# Phase 11b — WASM Renderer Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the `ferrum-wasm` crate — a WASM-compiled GPU renderer that consumes the SceneGraph IR produced by 11a and renders it to a browser `<canvas>`, plus CSS text overlays, self-contained `.save("chart.html")`, and `.save("chart.json")`. This is static rendering only (no selections, no zoom/pan, no anywidget — those are 11c).

**Architecture:** The `ferrum-wasm` crate depends on `ferrum-scene` (shared IR types) and compiles to `wasm32-unknown-unknown` via `wasm-pack`. It uses wgpu (WebGPU + WebGL2 fallback) for GPU rendering and lyon for path tessellation. Text is rendered as CSS overlays (`<div>` positioned absolutely). A JS ESM glue module bridges the DOM and WASM. Python-side, `save_chart` gains `"html"` and `"json"` format support by calling `render_interactive` (11a's SceneGraph JSON binding) and assembling a self-contained HTML bundle.

**Tech Stack:**
- Rust: `ferrum-scene` (existing), `ferrum-wasm` (new — wgpu, wasm-bindgen, web-sys, js-sys, lyon)
- JavaScript: ESM glue module (`ferrum-interactive.js`)
- CSS: Text overlay positioning (`ferrum-interactive.css`)
- Python: `_html.py` (new — HTML template assembly), updated `display.py` dispatcher
- Build: `wasm-pack` (target web), `maturin develop`

**Spec:** `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §5 (WASM renderer), §8 (error handling), §10.5 (save_chart), §11 (packaging).

**Prerequisite:** Phase 11a is complete. `render_interactive` PyO3 binding returns SceneGraph JSON. All golden SVGs pass.

---

## File map

### New files

| File | Purpose |
|---|---|
| `crates/ferrum-wasm/Cargo.toml` | Crate manifest — wgpu, wasm-bindgen, web-sys, js-sys, lyon, ferrum-scene, console-error-panic-hook |
| `crates/ferrum-wasm/src/lib.rs` | wasm-bindgen entry point: `WasmRenderer` exported to JS |
| `crates/ferrum-wasm/src/error.rs` | `WasmRenderError` enum (spec §8) |
| `crates/ferrum-wasm/src/gpu.rs` | wgpu initialization, surface/device/queue setup, WebGL2 fallback |
| `crates/ferrum-wasm/src/pipelines.rs` | `RenderPipelines` — instanced circle, instanced rect, mesh, textured |
| `crates/ferrum-wasm/src/shaders/circle.wgsl` | SDF circle vertex + fragment shader |
| `crates/ferrum-wasm/src/shaders/rect.wgsl` | SDF rounded rect vertex + fragment shader |
| `crates/ferrum-wasm/src/shaders/mesh.wgsl` | Lyon-tessellated mesh vertex + fragment shader |
| `crates/ferrum-wasm/src/shaders/textured.wgsl` | Textured quad vertex + fragment shader |
| `crates/ferrum-wasm/src/scene_load.rs` | Deserialize SceneGraph JSON, flatten Groups, build GPU buffers |
| `crates/ferrum-wasm/src/tessellate.rs` | Lyon tessellation: Line, Path, Polygon, Polyline → triangle mesh |
| `crates/ferrum-wasm/src/render.rs` | Frame rendering: draw calls per batch, clear color, viewport |
| `crates/ferrum-wasm/src/text.rs` | Extract Text nodes, produce JS-consumable text element descriptors |
| `src/ferrum/_wasm/` | Directory for WASM build output (created by wasm-pack) |
| `src/ferrum/_wasm/.gitkeep` | Placeholder so the directory survives git (actual artifacts gitignored) |
| `src/ferrum/_html.py` | HTML template assembly for `save("chart.html")` |

### Modified files

| File | Change |
|---|---|
| `Cargo.toml` (workspace root) | Add `crates/ferrum-wasm` to `members`; add wgpu, wasm-bindgen, web-sys, js-sys, lyon workspace deps |
| `pyproject.toml` | Add `[tool.maturin] include` for WASM artifacts |
| `src/ferrum/display.py` | Wire `"html"` and `"json"` formats in `save_chart` |
| `.gitignore` | Add `src/ferrum/_wasm/*.wasm`, `src/ferrum/_wasm/*.js` (build artifacts) |

### Unchanged files

All ferrum-core Rust code (render/, transform/, spec/, layout/, scale/). All ferrum-scene code. All existing Python modules except `display.py`. All test infrastructure. All golden SVGs.

---

## Task 11b1: Crate scaffold + wgpu initialization

Create the `ferrum-wasm` crate, configure wasm-pack build, initialize wgpu with WebGL2 fallback, and verify the WASM module loads in a browser.

**Files:**
- Create: `crates/ferrum-wasm/Cargo.toml`
- Create: `crates/ferrum-wasm/src/lib.rs`
- Create: `crates/ferrum-wasm/src/error.rs`
- Create: `crates/ferrum-wasm/src/gpu.rs`
- Modify: `Cargo.toml` (workspace root)

### Steps

- [ ] **Step 1: Install prerequisites**

Ensure `wasm-pack` and the WASM target are installed:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --version 0.13.1
```

Verify:

```bash
rustup target list --installed | grep wasm32
wasm-pack --version
```

Expected: `wasm32-unknown-unknown` listed, `wasm-pack 0.13.1`.

- [ ] **Step 2: Add ferrum-wasm to workspace and add workspace deps**

In the workspace root `Cargo.toml`, add `"crates/ferrum-wasm"` to the `members` array and add new workspace dependencies:

```toml
[workspace]
resolver = "2"
members = ["crates/ferrum-core", "crates/ferrum-scene", "crates/ferrum-wasm"]
```

Add to `[workspace.dependencies]`:

```toml
# Phase 11b — WASM renderer GPU backend.
# wgpu 24.x: WebGPU + WebGL2 fallback. The `webgl` feature enables the
# OpenGL ES / WebGL2 backend required for browsers without WebGPU support.
wgpu = { version = "24", features = ["webgl"] }
# wasm-bindgen / web-sys / js-sys for JS↔Rust interop in WASM.
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = [
    "Window", "Document", "Element", "HtmlCanvasElement",
    "console", "Performance",
] }
js-sys = "0.3"
# lyon: 2D path tessellation for GPU rendering of curves, areas, polygons.
# Pure Rust, compiles to WASM.
lyon = "1"
```

- [ ] **Step 3: Create ferrum-wasm Cargo.toml**

```toml
[package]
name = "ferrum-wasm"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[lib]
crate-type = ["cdylib"]

[dependencies]
ferrum-scene   = { path = "../ferrum-scene" }
wgpu           = { workspace = true }
wasm-bindgen   = { workspace = true }
wasm-bindgen-futures = { workspace = true }
web-sys        = { workspace = true }
js-sys         = { workspace = true }
lyon           = { workspace = true }
serde          = { workspace = true }
serde_json     = { workspace = true }
console-error-panic-hook = "0.1"
bytemuck       = { version = "1", features = ["derive"] }

[lints.clippy]
unwrap_used = "deny"
```

**Key points:**
- `crate-type = ["cdylib"]` is required for wasm-pack to produce a WASM module.
- `#[deny(clippy::unwrap_used)]` enforced at crate level per spec §8.1.
- No PyO3 dependency — this crate compiles only to `wasm32-unknown-unknown`.

- [ ] **Step 4: Create error.rs**

Write `crates/ferrum-wasm/src/error.rs`:

```rust
use wasm_bindgen::prelude::*;

/// All errors in the WASM renderer. Propagated to JS via
/// wasm_bindgen's Result → JsValue conversion.
#[derive(Debug, Clone)]
pub enum WasmRenderError {
    /// GPU adapter or device not available.
    GpuInit(String),
    /// GPU context lost mid-render.
    ContextLost,
    /// SceneGraph JSON failed to deserialize.
    SceneDeserialization(String),
    /// Texture upload failed (e.g., image too large).
    TextureUpload(String),
    /// Shader compilation failed.
    ShaderCompilation(String),
}

impl std::fmt::Display for WasmRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuInit(s) => write!(f, "GPU init failed: {s}"),
            Self::ContextLost => write!(f, "GPU context lost"),
            Self::SceneDeserialization(s) => write!(f, "scene deserialization: {s}"),
            Self::TextureUpload(s) => write!(f, "texture upload: {s}"),
            Self::ShaderCompilation(s) => write!(f, "shader compilation: {s}"),
        }
    }
}

impl std::error::Error for WasmRenderError {}

impl From<WasmRenderError> for JsValue {
    fn from(e: WasmRenderError) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
}
```

- [ ] **Step 5: Create gpu.rs — wgpu initialization**

Write `crates/ferrum-wasm/src/gpu.rs`:

```rust
use wgpu::{Adapter, Device, Queue, Surface, SurfaceConfiguration, TextureFormat};
use web_sys::HtmlCanvasElement;

use crate::error::WasmRenderError;

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub format: TextureFormat,
}

/// Initialize wgpu from an HTML canvas element.
///
/// Attempts WebGPU first, falls back to WebGL2 (via wgpu's `webgl` feature).
/// Returns Err if neither backend is available.
pub async fn init_gpu(canvas: HtmlCanvasElement) -> Result<GpuContext, WasmRenderError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
        ..Default::default()
    });

    let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    let surface = instance
        .create_surface(surface_target)
        .map_err(|e| WasmRenderError::GpuInit(format!("create_surface: {e}")))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| WasmRenderError::GpuInit(
            "no suitable GPU adapter found (WebGPU and WebGL2 both unavailable)".into(),
        ))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("ferrum-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        }, None)
        .await
        .map_err(|e| WasmRenderError::GpuInit(format!("request_device: {e}")))?;

    let width = canvas.width();
    let height = canvas.height();
    let format = surface
        .get_capabilities(&adapter)
        .formats
        .first()
        .copied()
        .unwrap_or(TextureFormat::Bgra8Unorm);

    let config = SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    Ok(GpuContext { device, queue, surface, config, format })
}
```

**Design notes:**
- `Limits::downlevel_webgl2_defaults()` ensures compatibility with WebGL2 — the lowest common denominator.
- `PowerPreference::LowPower` avoids spinning up a discrete GPU for chart rendering.
- `CompositeAlphaMode::PreMultiplied` enables transparency compositing with the CSS overlay layer.
- Canvas width/height are read directly from the element; the caller (JS) sets them before calling init.

- [ ] **Step 6: Create lib.rs — WasmRenderer stub**

Write `crates/ferrum-wasm/src/lib.rs`:

```rust
mod error;
mod gpu;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::error::WasmRenderError;
use crate::gpu::GpuContext;

/// The WASM renderer entry point, exported to JavaScript.
///
/// Each chart on the page gets its own `WasmRenderer` instance — no
/// global mutable state.
#[wasm_bindgen]
pub struct WasmRenderer {
    gpu: GpuContext,
    scene: Option<ferrum_scene::SceneGraph>,
}

#[wasm_bindgen]
impl WasmRenderer {
    /// Create a new renderer attached to the given canvas element.
    ///
    /// Call from JS: `const renderer = await WasmRenderer.new(canvas);`
    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement) -> Result<WasmRenderer, JsValue> {
        // Set panic hook for better WASM stack traces in console
        console_error_panic_hook::set_once();

        let gpu = gpu::init_gpu(canvas)
            .await
            .map_err(WasmRenderError::from)?;

        Ok(WasmRenderer { gpu, scene: None })
    }

    /// Load a SceneGraph from JSON and render the first frame.
    ///
    /// Returns a JSON array of text elements for the CSS overlay layer.
    /// Each element: `{ "x": f64, "y": f64, "content": str, "style": {...} }`
    #[wasm_bindgen(js_name = "loadScene")]
    pub fn load_scene(&mut self, scene_json: &str) -> Result<JsValue, JsValue> {
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(scene_json)
            .map_err(|e| WasmRenderError::SceneDeserialization(e.to_string()))?;

        // Extract text elements for CSS overlay (returned to JS)
        let text_elements = self.extract_text_elements(&scene);

        self.scene = Some(scene);

        // TODO (11b2): Build GPU buffers and render first frame

        serde_json::to_string(&text_elements)
            .map(|s| JsValue::from_str(&s))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Render a frame to the canvas.
    #[wasm_bindgen(js_name = "renderFrame")]
    pub fn render_frame(&self) -> Result<(), JsValue> {
        // TODO (11b2): Execute draw calls
        Ok(())
    }

    /// Resize the canvas and reconfigure the surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.config.width = width.max(1);
        self.gpu.config.height = height.max(1);
        self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
    }
}

impl WasmRenderer {
    fn extract_text_elements(
        &self,
        scene: &ferrum_scene::SceneGraph,
    ) -> Vec<TextElement> {
        // Stub — filled in 11b3
        Vec::new()
    }
}

#[derive(serde::Serialize)]
struct TextElement {
    x: f64,
    y: f64,
    content: String,
    font_size: f64,
    font_weight: String,
    font_family: String,
    anchor: String,
    baseline: String,
    angle: f64,
    color: String,
    opacity: f64,
}
```

(`console-error-panic-hook` is already in `Cargo.toml` from Step 3.)

- [ ] **Step 7: Verify WASM compilation**

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum
wasm-pack build crates/ferrum-wasm --target web --dev --out-dir ../../src/ferrum/_wasm/
```

Expected output:
- `src/ferrum/_wasm/ferrum_wasm_bg.wasm` — the WASM binary
- `src/ferrum/_wasm/ferrum_wasm.js` — the JS glue (ESM)
- `src/ferrum/_wasm/ferrum_wasm.d.ts` — TypeScript declarations
- `src/ferrum/_wasm/package.json` — npm package metadata (ignored)

If the build fails, common issues:
- Missing `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- wgpu feature compatibility: verify `webgl` feature is spelled correctly
- `wasm-pack` version too old: install 0.13.x+

- [ ] **Step 8: Create minimal browser test page**

Create `crates/ferrum-wasm/test.html` (NOT shipped in the wheel — dev-only):

```html
<!DOCTYPE html>
<html>
<head><title>ferrum-wasm init test</title></head>
<body>
  <canvas id="c" width="600" height="400" style="border:1px solid #ccc"></canvas>
  <pre id="log"></pre>
  <script type="module">
    import init, { WasmRenderer } from '../../src/ferrum/_wasm/ferrum_wasm.js';
    const log = document.getElementById('log');
    try {
      await init();
      log.textContent += 'WASM loaded OK\n';
      const canvas = document.getElementById('c');
      const renderer = await new WasmRenderer(canvas);
      log.textContent += 'WasmRenderer created OK\n';
      log.textContent += 'GPU init SUCCESS\n';
    } catch (e) {
      log.textContent += 'ERROR: ' + e + '\n';
    }
  </script>
</body>
</html>
```

**Manual verification:**

1. Start a local HTTP server (WASM requires HTTP, not file://):
   ```bash
   cd /Users/chrissantiago/Dropbox/GitHub/ferrum
   python3 -m http.server 8000
   ```
2. Open `http://localhost:8000/crates/ferrum-wasm/test.html` in Chrome/Firefox.
3. Expected in the page: `WASM loaded OK`, `WasmRenderer created OK`, `GPU init SUCCESS`.
4. No red errors in the browser console.

- [ ] **Step 9: Verify ferrum-core and ferrum-scene still compile**

The workspace addition must not break existing crates:

```bash
source ~/.cargo/env
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-scene -p ferrum-core
```

Expected: all existing tests pass unchanged.

- [ ] **Step 10: Commit**

```
feat(wasm): add ferrum-wasm crate scaffold with wgpu init and WebGL2 fallback
```

---

## Task 11b2: Mark rendering pipelines — SDF shaders, lyon tessellation

Build the four GPU pipelines (instanced circle, instanced rect, mesh, textured quad), WGSL shaders, lyon tessellation, and scene-to-GPU-buffer translation. After this task, `WasmRenderer.loadScene(json)` renders geometric primitives to the canvas.

**Files:**
- Create: `crates/ferrum-wasm/src/pipelines.rs`
- Create: `crates/ferrum-wasm/src/shaders/circle.wgsl`
- Create: `crates/ferrum-wasm/src/shaders/rect.wgsl`
- Create: `crates/ferrum-wasm/src/shaders/mesh.wgsl`
- Create: `crates/ferrum-wasm/src/shaders/textured.wgsl`
- Create: `crates/ferrum-wasm/src/scene_load.rs`
- Create: `crates/ferrum-wasm/src/tessellate.rs`
- Create: `crates/ferrum-wasm/src/render.rs`
- Modify: `crates/ferrum-wasm/src/lib.rs`

### GPU rendering strategy (from spec §5.1)

| SceneNode variant | Pipeline | Strategy |
|---|---|---|
| `Circle` | `instanced_circle` | Instanced quad + SDF circle in fragment shader. One draw call for N circles. |
| `Rect` | `instanced_rect` | Instanced quad + SDF rounded rect in fragment shader. Corner radius via SDF. |
| `Line`, `Path`, `Polygon`, `Polyline` | `mesh` | CPU tessellation via lyon → triangle mesh → single draw call. |
| `Image` | `textured` | Upload image bytes as GPU texture, draw positioned quad. |
| `Text` | CSS overlay | Not drawn by GPU — handled in 11b3 via JS/CSS. |
| `Group` | Flatten | Groups are SVG-specific wrappers. Flatten at load time: recurse into `children`, discard `attrs` (WASM reads `stroke_cap`/`stroke_join` from `MarkBatch` fields). |
| `Raw` | Skip | Raw nodes contain SVG-specific content (legend colorbar gradients). Log a console warning and skip. Typed gradient representation is 11c/11d scope. |

### Steps

- [ ] **Step 1: Create WGSL shaders**

Create `crates/ferrum-wasm/src/shaders/` directory.

**circle.wgsl** — SDF instanced circle:

```wgsl
// Vertex: unit quad [-1,1] × [-1,1], instanced per circle.
// Per-instance data: center (x,y), radius, color (RGBA), opacity.

struct Uniforms {
    viewport: vec2<f32>,  // canvas width, height
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,        // unit quad corner
    @location(1) center: vec2<f32>,          // instance: circle center (px)
    @location(2) radius: f32,                // instance: circle radius (px)
    @location(3) fill_color: vec4<f32>,      // instance: RGBA [0..1]
    @location(4) stroke_color: vec4<f32>,    // instance: stroke RGBA
    @location(5) stroke_width: f32,          // instance: stroke width (px)
    @location(6) opacity: f32,               // instance: overall opacity
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,       // [-1,1] within the quad
    @location(1) fill_color: vec4<f32>,
    @location(2) stroke_color: vec4<f32>,
    @location(3) stroke_width: f32,
    @location(4) radius: f32,
    @location(5) opacity: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let extent = in.radius + in.stroke_width + 1.0; // +1 for AA
    let px = in.center + in.quad_pos * extent;

    // Convert pixel coords to clip space: [0,w]×[0,h] → [-1,1]×[-1,1]
    // Note: Y is flipped (pixel Y grows down, clip Y grows up)
    let ndc = vec2<f32>(
        px.x / u.viewport.x * 2.0 - 1.0,
        1.0 - px.y / u.viewport.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.local_pos = in.quad_pos * extent;
    out.fill_color = in.fill_color;
    out.stroke_color = in.stroke_color;
    out.stroke_width = in.stroke_width;
    out.radius = in.radius;
    out.opacity = in.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.local_pos);
    // SDF: negative inside, positive outside
    let sdf = dist - in.radius;

    // Anti-aliased fill
    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, sdf);
    var color = in.fill_color * fill_alpha;

    // Stroke ring
    if in.stroke_width > 0.0 {
        let stroke_sdf = abs(sdf + in.stroke_width * 0.5) - in.stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_sdf);
        color = mix(color, in.stroke_color, stroke_alpha);
    }

    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
```

**rect.wgsl** — SDF instanced rounded rect:

```wgsl
struct Uniforms {
    viewport: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) rect_pos: vec2<f32>,        // top-left corner (px)
    @location(2) rect_size: vec2<f32>,        // width, height (px)
    @location(3) corner_radius: f32,
    @location(4) fill_color: vec4<f32>,
    @location(5) stroke_color: vec4<f32>,
    @location(6) stroke_width: f32,
    @location(7) opacity: f32,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) corner_radius: f32,
    @location(3) fill_color: vec4<f32>,
    @location(4) stroke_color: vec4<f32>,
    @location(5) stroke_width: f32,
    @location(6) opacity: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pad = in.stroke_width + 1.0;
    let center = in.rect_pos + in.rect_size * 0.5;
    let half = in.rect_size * 0.5 + pad;
    let px = center + in.quad_pos * half;

    let ndc = vec2<f32>(
        px.x / u.viewport.x * 2.0 - 1.0,
        1.0 - px.y / u.viewport.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.local_pos = in.quad_pos * half;
    out.half_size = in.rect_size * 0.5;
    out.corner_radius = in.corner_radius;
    out.fill_color = in.fill_color;
    out.stroke_color = in.stroke_color;
    out.stroke_width = in.stroke_width;
    out.opacity = in.opacity;
    return out;
}

// SDF for rounded rectangle
fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sdf = sdf_rounded_rect(in.local_pos, in.half_size, in.corner_radius);
    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, sdf);
    var color = in.fill_color * fill_alpha;

    if in.stroke_width > 0.0 {
        let stroke_sdf = abs(sdf + in.stroke_width * 0.5) - in.stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_sdf);
        color = mix(color, in.stroke_color, stroke_alpha);
    }

    color.a *= in.opacity;
    if color.a < 0.001 { discard; }
    return color;
}
```

**mesh.wgsl** — Lyon-tessellated triangles (fill + stroke):

```wgsl
struct Uniforms {
    viewport: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,    // pixel coordinates
    @location(1) color: vec4<f32>,       // per-vertex RGBA
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let ndc = vec2<f32>(
        in.position.x / u.viewport.x * 2.0 - 1.0,
        1.0 - in.position.y / u.viewport.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.color.a < 0.001 { discard; }
    return in.color;
}
```

**textured.wgsl** — Image quads:

```wgsl
struct Uniforms {
    viewport: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,    // pixel coords of quad corner
    @location(1) tex_coord: vec2<f32>,   // UV coordinates [0,1]
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let ndc = vec2<f32>(
        in.position.x / u.viewport.x * 2.0 - 1.0,
        1.0 - in.position.y / u.viewport.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.tex_coord = in.tex_coord;
    return out;
}

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coord);
}
```

- [ ] **Step 2: Create pipelines.rs**

Write `crates/ferrum-wasm/src/pipelines.rs`. This module creates the four render pipelines from the WGSL shaders.

```rust
use wgpu::{Device, RenderPipeline, TextureFormat, ShaderModule};

use crate::error::WasmRenderError;

pub struct RenderPipelines {
    pub instanced_circle: RenderPipeline,
    pub instanced_rect: RenderPipeline,
    pub mesh: RenderPipeline,
    pub textured: RenderPipeline,
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl RenderPipelines {
    pub fn new(device: &Device, format: TextureFormat) -> Result<Self, WasmRenderError> {
        let circle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("circle.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/circle.wgsl").into()),
        });
        let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rect.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rect.wgsl").into()),
        });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });
        let textured_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/textured.wgsl").into()),
        });

        // Shared uniform bind group layout (viewport size)
        let uniform_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        );

        // Texture bind group layout (for Image nodes)
        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        );

        // Build each pipeline with appropriate vertex layouts
        // (detailed vertex buffer layouts for each pipeline — see below)
        let instanced_circle = Self::build_circle_pipeline(
            device, &circle_shader, format, &uniform_bind_group_layout,
        );
        let instanced_rect = Self::build_rect_pipeline(
            device, &rect_shader, format, &uniform_bind_group_layout,
        );
        let mesh = Self::build_mesh_pipeline(
            device, &mesh_shader, format, &uniform_bind_group_layout,
        );
        let textured = Self::build_textured_pipeline(
            device, &textured_shader, format,
            &uniform_bind_group_layout, &texture_bind_group_layout,
        );

        Ok(Self {
            instanced_circle,
            instanced_rect,
            mesh,
            textured,
            uniform_bind_group_layout,
            texture_bind_group_layout,
        })
    }

    // Each build_*_pipeline function creates a wgpu::RenderPipeline with:
    // - The appropriate vertex buffer layout (quad vertices + per-instance data)
    // - Alpha blending enabled (SrcAlpha, OneMinusSrcAlpha)
    // - No depth/stencil (2D rendering, painter's algorithm via draw order)
    //
    // Implementation pattern (same for each variant):
    fn build_circle_pipeline(
        device: &Device, shader: &ShaderModule, format: TextureFormat,
        uniform_bgl: &wgpu::BindGroupLayout,
    ) -> RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("circle_pl"),
            bind_group_layouts: &[uniform_bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("circle"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                // Vertex buffer 0: unit quad (4 vertices)
                // Vertex buffer 1: per-instance data (step_mode: Instance)
                buffers: &[
                    // Buffer 0: quad vertex positions
                    wgpu::VertexBufferLayout {
                        array_stride: 8, // 2 × f32
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Buffer 1: per-instance circle data
                    wgpu::VertexBufferLayout {
                        // center(2) + radius(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1)
                        array_stride: 13 * 4, // 13 floats × 4 bytes
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { offset: 0,  shader_location: 1, format: wgpu::VertexFormat::Float32x2 }, // center
                            wgpu::VertexAttribute { offset: 8,  shader_location: 2, format: wgpu::VertexFormat::Float32 },   // radius
                            wgpu::VertexAttribute { offset: 12, shader_location: 3, format: wgpu::VertexFormat::Float32x4 }, // fill_color
                            wgpu::VertexAttribute { offset: 28, shader_location: 4, format: wgpu::VertexFormat::Float32x4 }, // stroke_color
                            wgpu::VertexAttribute { offset: 44, shader_location: 5, format: wgpu::VertexFormat::Float32 },   // stroke_width
                            wgpu::VertexAttribute { offset: 48, shader_location: 6, format: wgpu::VertexFormat::Float32 },   // opacity
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    // build_rect_pipeline, build_mesh_pipeline, build_textured_pipeline
    // follow the same pattern with their respective vertex layouts.
    // Mesh uses TriangleList (not TriangleStrip) with index buffer.
    // Textured uses TriangleStrip with texture bind group.
    // ... (full implementations follow the same structure)
}
```

Each `build_*_pipeline` function follows the pattern above. The mesh pipeline uses `TriangleList` topology (indexed) since lyon outputs triangles. The textured pipeline adds the `texture_bind_group_layout` as a second bind group.

- [ ] **Step 3: Create tessellate.rs — lyon tessellation**

Write `crates/ferrum-wasm/src/tessellate.rs`. This module converts Line, Path, Polygon, and Polyline SceneNodes into triangle meshes consumable by the mesh pipeline.

```rust
use ferrum_scene::*;
use lyon::math::point;
use lyon::path::Path as LyonPath;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions,
    StrokeTessellator, VertexBuffers,
};

/// A mesh vertex: position + color (matching mesh.wgsl layout).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// Tessellate a SceneNode::Line into a stroked line mesh.
pub fn tessellate_line(
    x1: f64, y1: f64, x2: f64, y2: f64,
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let color = color_to_f32_array(&style.color, style.opacity);
    let mut builder = LyonPath::builder();
    builder.begin(point(x1 as f32, y1 as f32));
    builder.line_to(point(x2 as f32, y2 as f32));
    builder.end(false);
    let path = builder.build();

    let mut opts = StrokeOptions::default();
    opts.line_width = style.width as f32;
    apply_stroke_options(&mut opts, style.stroke_cap, style.stroke_join);

    let mut tessellator = StrokeTessellator::new();
    let _ = tessellator.tessellate_path(
        &path,
        &opts,
        &mut BuffersBuilder::new(buffers, |pos: lyon::tessellation::StrokeVertex| {
            MeshVertex { position: pos.position().to_array(), color }
        }),
    );
}

/// Tessellate a SceneNode::Path into filled and/or stroked mesh.
pub fn tessellate_path(
    commands: &[PathCmd],
    style: &FillStroke,
    closed: bool,
    stroke_cap: Option<StrokeCap>,
    stroke_join: Option<StrokeJoin>,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    let path = pathcmds_to_lyon(commands, closed);

    // Fill
    if let Some(fill) = &style.fill {
        let color = color_to_f32_array(fill, style.opacity);
        let mut tessellator = FillTessellator::new();
        let _ = tessellator.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(buffers, |pos: lyon::tessellation::FillVertex| {
                MeshVertex { position: pos.position().to_array(), color }
            }),
        );
    }

    // Stroke
    if let Some(stroke) = &style.stroke {
        let color = color_to_f32_array(stroke, style.opacity);
        let mut opts = StrokeOptions::default();
        opts.line_width = style.stroke_width as f32;
        apply_stroke_options(&mut opts, stroke_cap, stroke_join);

        let mut tessellator = StrokeTessellator::new();
        let _ = tessellator.tessellate_path(
            &path,
            &opts,
            &mut BuffersBuilder::new(buffers, |pos: lyon::tessellation::StrokeVertex| {
                MeshVertex { position: pos.position().to_array(), color }
            }),
        );
    }
}

/// Tessellate a SceneNode::Polyline.
pub fn tessellate_polyline(
    points: &[(f64, f64)],
    style: &StrokeStyle,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    if points.len() < 2 { return; }
    let color = color_to_f32_array(&style.color, style.opacity);
    let mut builder = LyonPath::builder();
    builder.begin(point(points[0].0 as f32, points[0].1 as f32));
    for p in &points[1..] {
        builder.line_to(point(p.0 as f32, p.1 as f32));
    }
    builder.end(false);
    let path = builder.build();

    let mut opts = StrokeOptions::default();
    opts.line_width = style.width as f32;
    apply_stroke_options(&mut opts, style.stroke_cap, style.stroke_join);

    let mut tessellator = StrokeTessellator::new();
    let _ = tessellator.tessellate_path(
        &path,
        &opts,
        &mut BuffersBuilder::new(buffers, |pos: lyon::tessellation::StrokeVertex| {
            MeshVertex { position: pos.position().to_array(), color }
        }),
    );
}

/// Tessellate a SceneNode::Polygon.
pub fn tessellate_polygon(
    points: &[[f64; 2]],
    style: &FillStroke,
    buffers: &mut VertexBuffers<MeshVertex, u32>,
) {
    if points.len() < 3 { return; }
    let mut builder = LyonPath::builder();
    builder.begin(point(points[0][0] as f32, points[0][1] as f32));
    for p in &points[1..] {
        builder.line_to(point(p[0] as f32, p[1] as f32));
    }
    builder.close();
    let path = builder.build();

    // Fill
    if let Some(fill) = &style.fill {
        let color = color_to_f32_array(fill, style.opacity);
        let mut tessellator = FillTessellator::new();
        let _ = tessellator.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(buffers, |pos: lyon::tessellation::FillVertex| {
                MeshVertex { position: pos.position().to_array(), color }
            }),
        );
    }

    // Stroke
    if let Some(stroke) = &style.stroke {
        let color = color_to_f32_array(stroke, style.opacity);
        let mut opts = StrokeOptions::default();
        opts.line_width = style.stroke_width as f32;

        let mut tessellator = StrokeTessellator::new();
        let _ = tessellator.tessellate_path(
            &path,
            &opts,
            &mut BuffersBuilder::new(buffers, |pos: lyon::tessellation::StrokeVertex| {
                MeshVertex { position: pos.position().to_array(), color }
            }),
        );
    }
}

// --- Internal helpers ---

fn pathcmds_to_lyon(cmds: &[PathCmd], closed: bool) -> LyonPath {
    let mut builder = LyonPath::builder();
    // Track current position manually — lyon's PathBuilder does not
    // expose current_position() as a public method in 1.x.
    let mut cur_x: f32 = 0.0;
    let mut cur_y: f32 = 0.0;
    for cmd in cmds {
        match cmd {
            PathCmd::MoveTo { x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.begin(point(cur_x, cur_y));
            }
            PathCmd::LineTo { x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.line_to(point(cur_x, cur_y));
            }
            PathCmd::HLineTo { x } => {
                // Horizontal line: keep current Y, move to new X.
                cur_x = *x as f32;
                builder.line_to(point(cur_x, cur_y));
            }
            PathCmd::VLineTo { y } => {
                // Vertical line: keep current X, move to new Y.
                cur_y = *y as f32;
                builder.line_to(point(cur_x, cur_y));
            }
            PathCmd::QuadTo { cx, cy, x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.quadratic_bezier_to(
                    point(*cx as f32, *cy as f32),
                    point(cur_x, cur_y),
                );
            }
            PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                builder.cubic_bezier_to(
                    point(*c1x as f32, *c1y as f32),
                    point(*c2x as f32, *c2y as f32),
                    point(cur_x, cur_y),
                );
            }
            PathCmd::ArcTo { rx, ry, rotation, large_arc, sweep, x, y } => {
                cur_x = *x as f32;
                cur_y = *y as f32;
                // lyon uses SvgArc for SVG-compatible arc segments
                builder.arc_to(
                    lyon::math::vector(*rx as f32, *ry as f32),
                    lyon::math::Angle::degrees(*rotation as f32),
                    lyon::path::ArcFlags {
                        large_arc: *large_arc,
                        sweep: *sweep,
                    },
                    point(cur_x, cur_y),
                );
            }
            PathCmd::Close => {
                builder.close();
                // Note: Close does not update cur_x/cur_y — after close,
                // the next MoveTo resets position. If no MoveTo follows,
                // the builder state is undefined, which is fine (end of path).
            }
        }
    }
    if closed && !matches!(cmds.last(), Some(PathCmd::Close)) {
        builder.close();
    } else if !closed {
        builder.end(false);
    }
    builder.build()
}

fn color_to_f32_array(c: &Color, opacity: f64) -> [f32; 4] {
    [
        c.r as f32 / 255.0,
        c.g as f32 / 255.0,
        c.b as f32 / 255.0,
        (c.a as f32 / 255.0) * opacity as f32,
    ]
}

fn apply_stroke_options(
    opts: &mut StrokeOptions,
    cap: Option<StrokeCap>,
    join: Option<StrokeJoin>,
) {
    if let Some(c) = cap {
        opts.start_cap = match c {
            StrokeCap::Butt => lyon::tessellation::LineCap::Butt,
            StrokeCap::Round => lyon::tessellation::LineCap::Round,
            StrokeCap::Square => lyon::tessellation::LineCap::Square,
        };
        opts.end_cap = opts.start_cap;
    }
    if let Some(j) = join {
        opts.line_join = match j {
            StrokeJoin::Miter => lyon::tessellation::LineJoin::Miter,
            StrokeJoin::Round => lyon::tessellation::LineJoin::Round,
            StrokeJoin::Bevel => lyon::tessellation::LineJoin::Bevel,
        };
    }
}
```

Add `bytemuck` to `Cargo.toml`:

```toml
bytemuck = { version = "1", features = ["derive"] }
```

- [ ] **Step 4: Create scene_load.rs — SceneGraph → GPU buffers**

Write `crates/ferrum-wasm/src/scene_load.rs`. This module deserializes the SceneGraph JSON, flattens Group nodes, and builds GPU buffers for each pipeline.

```rust
use ferrum_scene::*;
use lyon::tessellation::VertexBuffers;

use crate::tessellate::{self, MeshVertex};

/// Per-instance data for the circle pipeline (matches circle.wgsl layout).
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

/// Per-instance data for the rect pipeline (matches rect.wgsl layout).
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

/// Collected GPU data from a SceneGraph, ready for buffer upload.
pub struct LoadedScene {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
    pub mesh_buffers: VertexBuffers<MeshVertex, u32>,
    pub text_elements: Vec<TextElementData>,
    pub background: Option<[f32; 4]>,
    pub width: f32,
    pub height: f32,
}

pub struct TextElementData {
    pub x: f64,
    pub y: f64,
    pub content: String,
    pub style: TextStyle,
}

/// Load a SceneGraph into GPU-ready data structures.
///
/// Flattens Group nodes (discards attrs — SVG-only concern).
/// Skips Raw nodes with a console warning.
/// Collects Text nodes for CSS overlay.
pub fn load_scene(scene: &SceneGraph) -> LoadedScene {
    let mut circles = Vec::new();
    let mut rects = Vec::new();
    let mut mesh = VertexBuffers::new();
    let mut texts = Vec::new();

    let background = scene.background.as_ref().map(|c| {
        [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, c.a as f32 / 255.0]
    });

    // Process title nodes
    collect_nodes(&scene.title, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);

    // Process panels
    for panel in &scene.panels {
        // Grid (behind marks)
        collect_nodes(&panel.grid, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);

        // Marks — z-ordered by batch order
        for batch in &panel.marks {
            collect_nodes(
                &batch.nodes,
                &mut circles, &mut rects, &mut mesh, &mut texts,
                batch.stroke_cap, batch.stroke_join,
            );
        }

        // Axes (on top of marks)
        collect_nodes(&panel.axes, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);

        // Strip titles
        collect_nodes(&panel.strip_title, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);

        // Annotations
        collect_nodes(&panel.annotations, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);
    }

    // Legend nodes
    collect_nodes(&scene.legend, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);

    // Decorations
    collect_nodes(&scene.decorations, &mut circles, &mut rects, &mut mesh, &mut texts, None, None);

    LoadedScene {
        circle_instances: circles,
        rect_instances: rects,
        mesh_buffers: mesh,
        text_elements: texts,
        background,
        width: scene.width as f32,
        height: scene.height as f32,
    }
}

fn collect_nodes(
    nodes: &[SceneNode],
    circles: &mut Vec<CircleInstance>,
    rects: &mut Vec<RectInstance>,
    mesh: &mut VertexBuffers<MeshVertex, u32>,
    texts: &mut Vec<TextElementData>,
    batch_cap: Option<StrokeCap>,
    batch_join: Option<StrokeJoin>,
) {
    for node in nodes {
        match node {
            SceneNode::Circle { cx, cy, r, style } => {
                circles.push(CircleInstance {
                    center: [*cx as f32, *cy as f32],
                    radius: *r as f32,
                    fill_color: fill_to_f32(style.fill.as_ref(), style.opacity),
                    stroke_color: fill_to_f32(style.stroke.as_ref(), style.opacity),
                    stroke_width: style.stroke_width as f32,
                    opacity: style.opacity as f32,
                });
            }
            SceneNode::Rect { x, y, w, h, style, corner_radius } => {
                rects.push(RectInstance {
                    position: [*x as f32, *y as f32],
                    size: [*w as f32, *h as f32],
                    corner_radius: *corner_radius as f32,
                    fill_color: fill_to_f32(style.fill.as_ref(), style.opacity),
                    stroke_color: fill_to_f32(style.stroke.as_ref(), style.opacity),
                    stroke_width: style.stroke_width as f32,
                    opacity: style.opacity as f32,
                });
            }
            SceneNode::Line { x1, y1, x2, y2, style } => {
                let mut style_with_caps = style.clone();
                if style_with_caps.stroke_cap.is_none() {
                    style_with_caps.stroke_cap = batch_cap;
                }
                if style_with_caps.stroke_join.is_none() {
                    style_with_caps.stroke_join = batch_join;
                }
                tessellate::tessellate_line(*x1, *y1, *x2, *y2, &style_with_caps, mesh);
            }
            SceneNode::Path { commands, style, closed } => {
                tessellate::tessellate_path(
                    commands, style, *closed, batch_cap, batch_join, mesh,
                );
            }
            SceneNode::Polyline { points, style } => {
                let mut style_with_caps = style.clone();
                if style_with_caps.stroke_cap.is_none() {
                    style_with_caps.stroke_cap = batch_cap;
                }
                if style_with_caps.stroke_join.is_none() {
                    style_with_caps.stroke_join = batch_join;
                }
                tessellate::tessellate_polyline(points, &style_with_caps, mesh);
            }
            SceneNode::Polygon { points, style } => {
                tessellate::tessellate_polygon(points, style, mesh);
            }
            SceneNode::Text { x, y, content, style } => {
                texts.push(TextElementData {
                    x: *x,
                    y: *y,
                    content: content.clone(),
                    style: style.clone(),
                });
            }
            SceneNode::Image { .. } => {
                // TODO: texture upload for Image nodes (rare in stat charts)
                web_sys::console::warn_1(
                    &"ferrum: Image nodes not yet supported in WASM renderer".into(),
                );
            }
            SceneNode::Group { children, .. } => {
                // Flatten: recurse into children, discard SVG-specific attrs.
                // Batch-level stroke_cap/stroke_join propagated from the MarkBatch.
                collect_nodes(children, circles, rects, mesh, texts, batch_cap, batch_join);
            }
            SceneNode::Raw { .. } => {
                // Raw SVG content (legend colorbar gradients).
                // Cannot render in GPU — skip with console warning.
                // Typed gradient representation is 11c/11d scope.
                web_sys::console::warn_1(
                    &"ferrum: Raw SVG node skipped (colorbar gradients not yet supported in WASM)".into(),
                );
            }
        }
    }
}

fn fill_to_f32(color: Option<&Color>, opacity: f64) -> [f32; 4] {
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
```

**Key design decisions:**
- Group nodes are flattened: recurse into `children`, discard `attrs`. The attrs (like `stroke-linecap`) are SVG-specific; the WASM renderer reads `stroke_cap`/`stroke_join` from `MarkBatch` fields, passed as `batch_cap`/`batch_join` parameters.
- Raw nodes are skipped with a `console.warn`. They contain SVG-specific content (colorbar `<defs>` + `<linearGradient>`). A typed gradient representation is out of scope for 11b.
- Image nodes are stubbed with a warning (rare in stat charts; texture pipeline is wired but uploads are deferred to 11b follow-up if needed).

- [ ] **Step 5: Create render.rs — frame rendering**

Write `crates/ferrum-wasm/src/render.rs`. This module uploads GPU buffers and executes draw calls.

```rust
use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;
use crate::pipelines::RenderPipelines;
use crate::scene_load::{CircleInstance, LoadedScene, RectInstance};
use crate::tessellate::MeshVertex;

/// GPU resources for a loaded scene.
pub struct GpuBuffers {
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub quad_vertex_buffer: wgpu::Buffer,
    pub circle_instance_buffer: Option<wgpu::Buffer>,
    pub circle_count: u32,
    pub rect_instance_buffer: Option<wgpu::Buffer>,
    pub rect_count: u32,
    pub mesh_vertex_buffer: Option<wgpu::Buffer>,
    pub mesh_index_buffer: Option<wgpu::Buffer>,
    pub mesh_index_count: u32,
}

// Unit quad vertices: 4 corners for TriangleStrip
const QUAD_VERTICES: [[f32; 2]; 4] = [
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0,  1.0],
];

impl GpuBuffers {
    pub fn from_loaded_scene(
        gpu: &GpuContext,
        pipelines: &RenderPipelines,
        scene: &LoadedScene,
    ) -> Self {
        let viewport = [scene.width, scene.height];
        let uniform_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&viewport),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms_bg"),
            layout: &pipelines.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let quad_vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let circle_instance_buffer = if scene.circle_instances.is_empty() {
            None
        } else {
            Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("circles"),
                contents: bytemuck::cast_slice(&scene.circle_instances),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };

        let rect_instance_buffer = if scene.rect_instances.is_empty() {
            None
        } else {
            Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rects"),
                contents: bytemuck::cast_slice(&scene.rect_instances),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };

        let (mesh_vertex_buffer, mesh_index_buffer) = if scene.mesh_buffers.vertices.is_empty() {
            (None, None)
        } else {
            (
                Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_verts"),
                    contents: bytemuck::cast_slice(&scene.mesh_buffers.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })),
                Some(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_idx"),
                    contents: bytemuck::cast_slice(&scene.mesh_buffers.indices),
                    usage: wgpu::BufferUsages::INDEX,
                })),
            )
        };

        GpuBuffers {
            uniform_buffer,
            uniform_bind_group,
            quad_vertex_buffer,
            circle_instance_buffer,
            circle_count: scene.circle_instances.len() as u32,
            rect_instance_buffer,
            rect_count: scene.rect_instances.len() as u32,
            mesh_vertex_buffer,
            mesh_index_buffer,
            mesh_index_count: scene.mesh_buffers.indices.len() as u32,
        }
    }
}

/// Render one frame to the surface.
pub fn render_frame(
    gpu: &GpuContext,
    pipelines: &RenderPipelines,
    buffers: &GpuBuffers,
    clear_color: Option<[f32; 4]>,
) -> Result<(), crate::error::WasmRenderError> {
    let output = gpu.surface
        .get_current_texture()
        .map_err(|e| crate::error::WasmRenderError::GpuInit(format!("get_current_texture: {e}")))?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bg = clear_color.unwrap_or([1.0, 1.0, 1.0, 1.0]);

    let mut encoder = gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("frame") },
    );

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: bg[0] as f64,
                        g: bg[1] as f64,
                        b: bg[2] as f64,
                        a: bg[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Draw mesh first (areas, paths — background fills)
        if let (Some(vb), Some(ib)) = (&buffers.mesh_vertex_buffer, &buffers.mesh_index_buffer) {
            pass.set_pipeline(&pipelines.mesh);
            pass.set_bind_group(0, &buffers.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buffers.mesh_index_count, 0, 0..1);
        }

        // Draw rects (bars, heatmap cells)
        if let Some(ib) = &buffers.rect_instance_buffer {
            pass.set_pipeline(&pipelines.instanced_rect);
            pass.set_bind_group(0, &buffers.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, buffers.quad_vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, ib.slice(..));
            pass.draw(0..4, 0..buffers.rect_count);
        }

        // Draw circles (points) — on top
        if let Some(ib) = &buffers.circle_instance_buffer {
            pass.set_pipeline(&pipelines.instanced_circle);
            pass.set_bind_group(0, &buffers.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, buffers.quad_vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, ib.slice(..));
            pass.draw(0..4, 0..buffers.circle_count);
        }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    output.present();
    Ok(())
}
```

**Draw order rationale:** mesh (areas, lines, polygons — background fills) → rects (bars) → circles (points on top). This matches the SVG z-order where marks in later `MarkBatch` entries render on top. A fully correct z-order implementation for interleaved batch types requires sorting draw calls by batch index; the simplified approach here (mesh → rect → circle) matches the common case (area behind line behind points) and can be refined in 11c if needed.

- [ ] **Step 6: Wire everything into lib.rs**

Update `crates/ferrum-wasm/src/lib.rs` to use the new modules and fully implement `load_scene` and `render_frame`:

```rust
mod error;
mod gpu;
mod pipelines;
mod scene_load;
mod tessellate;
mod render;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::error::WasmRenderError;
use crate::gpu::GpuContext;
use crate::pipelines::RenderPipelines;
use crate::render::GpuBuffers;
use crate::scene_load::LoadedScene;

#[wasm_bindgen]
pub struct WasmRenderer {
    gpu: GpuContext,
    pipelines: RenderPipelines,
    loaded: Option<LoadedSceneGpu>,
}

struct LoadedSceneGpu {
    scene_data: LoadedScene,
    gpu_buffers: GpuBuffers,
}

#[wasm_bindgen]
impl WasmRenderer {
    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement) -> Result<WasmRenderer, JsValue> {
        console_error_panic_hook::set_once();

        let gpu = gpu::init_gpu(canvas)
            .await
            .map_err(|e| JsValue::from(e))?;

        let pipelines = RenderPipelines::new(&gpu.device, gpu.format)
            .map_err(|e| JsValue::from(e))?;

        Ok(WasmRenderer { gpu, pipelines, loaded: None })
    }

    /// Load a SceneGraph from JSON, build GPU buffers, render first frame.
    ///
    /// Returns a JSON array of text elements for the CSS overlay layer.
    #[wasm_bindgen(js_name = "loadScene")]
    pub fn load_scene(&mut self, scene_json: &str) -> Result<String, JsValue> {
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(scene_json)
            .map_err(|e| JsValue::from(WasmRenderError::SceneDeserialization(e.to_string())))?;

        let scene_data = scene_load::load_scene(&scene);

        // Build text element descriptors for JS
        let text_json = self.build_text_json(&scene_data);

        let gpu_buffers = GpuBuffers::from_loaded_scene(
            &self.gpu, &self.pipelines, &scene_data,
        );

        let clear_color = scene_data.background;

        self.loaded = Some(LoadedSceneGpu { scene_data, gpu_buffers });

        // Render first frame
        if let Some(ref loaded) = self.loaded {
            render::render_frame(
                &self.gpu, &self.pipelines, &loaded.gpu_buffers, clear_color,
            ).map_err(|e| JsValue::from(e))?;
        }

        Ok(text_json)
    }

    #[wasm_bindgen(js_name = "renderFrame")]
    pub fn render_frame(&self) -> Result<(), JsValue> {
        if let Some(ref loaded) = self.loaded {
            let clear_color = loaded.scene_data.background;
            render::render_frame(
                &self.gpu, &self.pipelines, &loaded.gpu_buffers, clear_color,
            ).map_err(|e| JsValue::from(e))?;
        }
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.config.width = width.max(1);
        self.gpu.config.height = height.max(1);
        self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
        // Re-render after resize
        let _ = self.render_frame();
    }
}

impl WasmRenderer {
    fn build_text_json(&self, scene_data: &LoadedScene) -> String {
        let elements: Vec<serde_json::Value> = scene_data.text_elements.iter().map(|t| {
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
        }).collect();
        serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
    }
}
```

- [ ] **Step 7: Rebuild WASM with all modules**

`bytemuck` and `console-error-panic-hook` were already added in 11b1 Step 3. Rebuild:

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum
wasm-pack build crates/ferrum-wasm --target web --dev --out-dir ../../src/ferrum/_wasm/
```

Expected: builds successfully. WASM binary in `src/ferrum/_wasm/`.

- [ ] **Step 8: Create integration test page**

Update `crates/ferrum-wasm/test.html` to load a real SceneGraph JSON:

```html
<!DOCTYPE html>
<html>
<head><title>ferrum-wasm render test</title></head>
<body>
  <div style="position:relative">
    <canvas id="c" width="600" height="400" style="border:1px solid #ccc"></canvas>
    <div id="overlay" style="position:absolute;top:0;left:0;pointer-events:none"></div>
  </div>
  <pre id="log"></pre>
  <script type="module">
    import init, { WasmRenderer } from '../../src/ferrum/_wasm/ferrum_wasm.js';
    const log = document.getElementById('log');
    try {
      await init();
      log.textContent += 'WASM loaded OK\n';

      const canvas = document.getElementById('c');
      const renderer = await new WasmRenderer(canvas);
      log.textContent += 'GPU init OK\n';

      // Load SceneGraph JSON (generated by: python -c "...")
      const resp = await fetch('./test_scene.json');
      const sceneJson = await resp.text();
      const textElements = JSON.parse(renderer.loadScene(sceneJson));
      log.textContent += `Rendered ${textElements.length} text elements\n`;

      // Place text overlay divs
      const overlay = document.getElementById('overlay');
      for (const t of textElements) {
        const div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.left = t.x + 'px';
        div.style.top = t.y + 'px';
        div.style.fontSize = t.fontSize + 'px';
        div.style.fontWeight = t.fontWeight;
        div.style.fontFamily = t.fontFamily;
        div.style.color = t.color;
        div.style.whiteSpace = 'nowrap';
        div.style.transform = `rotate(${t.angle}deg)`;
        div.style.textAnchor = t.anchor; // CSS text-align mapped below
        div.textContent = t.content;

        // Map anchor → CSS text-align
        if (t.anchor === 'center') div.style.textAlign = 'center';
        else if (t.anchor === 'end') div.style.textAlign = 'right';

        overlay.appendChild(div);
      }

      log.textContent += 'SUCCESS: chart rendered\n';
    } catch (e) {
      log.textContent += 'ERROR: ' + e + '\n';
      console.error(e);
    }
  </script>
</body>
</html>
```

Generate a test SceneGraph JSON file:

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum
unset CONDA_PREFIX && uv run --no-sync python -c "
import polars as pl
import ferrum as fm
from ferrum._core import render_interactive, ChartSpec

# Simple scatter
df = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 50.0, 30.0, 80.0, 60.0]})
chart = fm.Chart(df).mark_point().encode(x='x', y='y')
spec = chart._build_spec()
batch = chart._build_batch()
json_str = render_interactive(spec, batch, viewport=(600.0, 400.0))
with open('crates/ferrum-wasm/test_scene.json', 'w') as f:
    f.write(json_str)
print('OK: test_scene.json written')
"
```

**Manual verification:**

1. `python3 -m http.server 8000` from the repo root.
2. Open `http://localhost:8000/crates/ferrum-wasm/test.html`.
3. Expected: 5 colored circles visible on the canvas at different positions. Axis tick labels visible as overlay text. No red errors in console.
4. Verify: background color matches theme default (white). Circles use the default mark color.

- [ ] **Step 9: Commit**

```
feat(wasm): add GPU rendering pipelines — SDF shaders, lyon tessellation, scene-to-canvas rendering
```

---

## Task 11b3: JS glue module + text overlay + HTML output

Create the JavaScript ESM glue module and CSS for standalone HTML output. After this task, `save("chart.html")` produces a self-contained HTML file that renders a chart.

**Files:**
- Create: `src/ferrum/_wasm/ferrum-interactive.js`
- Create: `src/ferrum/_wasm/ferrum-interactive.css`
- Create: `src/ferrum/_html.py`
- Modify: `src/ferrum/_wasm/.gitkeep` → remove (replaced by actual files)

### Steps

- [ ] **Step 1: Create ferrum-interactive.css**

Write `src/ferrum/_wasm/ferrum-interactive.css`:

```css
/* ferrum interactive chart — CSS overlay for text elements */

.ferrum-root {
  position: relative;
  display: inline-block;
}

.ferrum-root canvas {
  display: block;
}

.ferrum-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: hidden;
}

.ferrum-text {
  position: absolute;
  white-space: nowrap;
  line-height: 1;
  /* Prevent text selection during pan/zoom (11c) */
  user-select: none;
  -webkit-user-select: none;
}

/* Text anchor mapping */
.ferrum-text[data-anchor="start"] {
  text-align: left;
}
.ferrum-text[data-anchor="center"] {
  text-align: center;
  transform-origin: center center;
}
.ferrum-text[data-anchor="end"] {
  text-align: right;
}

/* Baseline mapping */
.ferrum-text[data-baseline="top"] {
  /* top of text box aligns to y coordinate */
}
.ferrum-text[data-baseline="middle"] {
  transform: translateY(-50%);
}
.ferrum-text[data-baseline="bottom"] {
  transform: translateY(-100%);
}
.ferrum-text[data-baseline="alphabetic"] {
  /* Default browser baseline — approximately 80% from top */
  transform: translateY(-0.8em);
}
```

- [ ] **Step 2: Create ferrum-interactive.js**

Write `src/ferrum/_wasm/ferrum-interactive.js`. This is the ESM module that bridges the DOM and WASM renderer.

```javascript
/**
 * ferrum-interactive.js — ESM glue module for the ferrum WASM renderer.
 *
 * Two modes:
 * 1. Standalone: called from inline <script> in save("chart.html") output.
 *    No `model` parameter — just render the scene.
 * 2. anywidget (11c): called with `model` for Jupyter bidirectional state.
 *
 * 11b ships standalone mode only.
 */

// The WASM init function and WasmRenderer class are imported from the
// wasm-pack-generated glue. In standalone HTML, they're loaded from the
// same bundle. In anywidget mode, the import path is rewritten.
let wasmInit, WasmRendererClass;

/**
 * Initialize the WASM module. Must be called once before creating renderers.
 * @param {Object} wasmModule - The wasm-pack generated module
 */
export async function initWasm(wasmModule) {
  wasmInit = wasmModule.default;
  WasmRendererClass = wasmModule.WasmRenderer;
}

/**
 * Render a ferrum chart into a container element.
 *
 * @param {Object} options
 * @param {HTMLElement} options.el - Container element
 * @param {string} options.sceneJson - SceneGraph JSON string
 * @param {number} options.width - Canvas width in pixels
 * @param {number} options.height - Canvas height in pixels
 * @param {Object} [options.model] - anywidget model (null for standalone)
 * @returns {Promise<WasmRenderer>} The renderer instance
 */
export async function renderChart({ el, sceneJson, width, height, model = null }) {
  // Create DOM structure
  const root = document.createElement('div');
  root.className = 'ferrum-root';
  root.style.width = width + 'px';
  root.style.height = height + 'px';

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  root.appendChild(canvas);

  const overlay = document.createElement('div');
  overlay.className = 'ferrum-overlay';
  root.appendChild(overlay);

  el.appendChild(root);

  // Initialize WASM renderer
  const renderer = await new WasmRendererClass(canvas);

  // Load scene and get text elements
  const textElementsJson = renderer.loadScene(sceneJson);
  const textElements = JSON.parse(textElementsJson);

  // Place text overlays
  placeTextOverlays(overlay, textElements);

  return renderer;
}

/**
 * Place text elements as positioned <div>s in the overlay container.
 */
function placeTextOverlays(container, textElements) {
  for (const t of textElements) {
    const div = document.createElement('div');
    div.className = 'ferrum-text';
    div.textContent = t.content;

    // Positioning
    div.style.left = t.x + 'px';
    div.style.top = t.y + 'px';

    // Typography
    div.style.fontSize = t.fontSize + 'px';
    div.style.fontWeight = t.fontWeight;
    div.style.fontFamily = t.fontFamily;
    div.style.color = t.color;

    // Anchor → CSS alignment
    div.dataset.anchor = t.anchor;
    if (t.anchor === 'center') {
      div.style.transform = `translateX(-50%)`;
    } else if (t.anchor === 'end') {
      div.style.transform = `translateX(-100%)`;
    }

    // Baseline
    div.dataset.baseline = t.baseline;
    applyBaseline(div, t.baseline, t.anchor);

    // Rotation (around the anchor point)
    if (Math.abs(t.angle) > 0.01) {
      const existing = div.style.transform || '';
      div.style.transform = existing + ` rotate(${t.angle}deg)`;
    }

    container.appendChild(div);
  }
}

/**
 * Apply baseline offset via transform.
 * Combined with any existing anchor transform.
 */
function applyBaseline(div, baseline, anchor) {
  let tx = '';
  if (anchor === 'center') tx = 'translateX(-50%)';
  else if (anchor === 'end') tx = 'translateX(-100%)';

  let ty = '';
  if (baseline === 'middle') ty = 'translateY(-50%)';
  else if (baseline === 'bottom') ty = 'translateY(-100%)';
  else if (baseline === 'alphabetic') ty = 'translateY(-0.8em)';
  // 'top' and custom baselines: no Y offset

  const combined = [tx, ty].filter(Boolean).join(' ');
  if (combined) {
    div.style.transform = combined;
  }
}

/**
 * anywidget entry point (11c).
 * Called by anywidget when the widget mounts.
 */
export async function render({ model, el }) {
  // 11b: standalone mode only. anywidget wiring is 11c.
  const sceneJson = model.get('scene_json');
  const width = model.get('width') || 600;
  const height = model.get('height') || 400;
  await renderChart({ el, sceneJson, width, height, model });
}
```

**Note:** This file is hand-authored, NOT generated by wasm-pack. The wasm-pack-generated files (`ferrum_wasm.js`, `ferrum_wasm_bg.wasm`) are separate. This file imports from them in the HTML template.

- [ ] **Step 3: Create _html.py — HTML template assembly**

Write `src/ferrum/_html.py`:

```python
"""Self-contained HTML bundle assembly for chart.save("chart.html")."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Directory containing WASM build artifacts
_WASM_DIR = Path(__file__).parent / "_wasm"


def _read_wasm_binary() -> bytes:
    """Read the compiled WASM binary.

    Raises
    ------
    FileNotFoundError
        If the WASM binary has not been built. This happens in development
        when ``wasm-pack build`` has not been run.
    """
    wasm_path = _WASM_DIR / "ferrum_wasm_bg.wasm"
    if not wasm_path.exists():
        raise FileNotFoundError(
            f"WASM binary not found at {wasm_path}. "
            "Run: wasm-pack build crates/ferrum-wasm --target web "
            "--out-dir ../../src/ferrum/_wasm/"
        )
    return wasm_path.read_bytes()


def _read_wasm_js_glue() -> str:
    """Read the wasm-pack-generated JS glue module."""
    js_path = _WASM_DIR / "ferrum_wasm.js"
    if not js_path.exists():
        raise FileNotFoundError(
            f"WASM JS glue not found at {js_path}. "
            "Run: wasm-pack build crates/ferrum-wasm --target web "
            "--out-dir ../../src/ferrum/_wasm/"
        )
    return js_path.read_text()


def _read_interactive_js() -> str:
    """Read the hand-authored ferrum-interactive.js module."""
    js_path = _WASM_DIR / "ferrum-interactive.js"
    if not js_path.exists():
        raise FileNotFoundError(f"ferrum-interactive.js not found at {js_path}.")
    return js_path.read_text()


def _read_interactive_css() -> str:
    """Read the ferrum-interactive.css styles."""
    css_path = _WASM_DIR / "ferrum-interactive.css"
    if not css_path.exists():
        raise FileNotFoundError(f"ferrum-interactive.css not found at {css_path}.")
    return css_path.read_text()


def build_html_bundle(
    scene_json: str,
    *,
    width: float = 600.0,
    height: float = 400.0,
    title: str = "Ferrum chart",
    embed_wasm: bool = True,
) -> str:
    """Build a self-contained HTML file that renders a chart via WASM.

    Parameters
    ----------
    scene_json : str
        SceneGraph JSON string (from ``render_interactive``).
    width : float
        Canvas width in pixels.
    height : float
        Canvas height in pixels.
    title : str
        HTML document title.
    embed_wasm : bool
        If True (default), base64-encode the WASM binary inline for
        single-file distribution. If False, the HTML references
        ``ferrum_wasm_bg.wasm`` as a sidecar file.

    Returns
    -------
    str
        Complete HTML document string.
    """
    css = _read_interactive_css()
    wasm_glue_js = _read_wasm_js_glue()
    interactive_js = _read_interactive_js()

    if embed_wasm:
        wasm_bytes = _read_wasm_binary()
        wasm_b64 = base64.b64encode(wasm_bytes).decode("ascii")
        wasm_init_code = f"""
        // Decode base64 WASM binary
        const wasmB64 = "{wasm_b64}";
        const wasmBytes = Uint8Array.from(atob(wasmB64), c => c.charCodeAt(0));
        await init(wasmBytes);
        """
    else:
        wasm_init_code = """
        await init();  // loads ferrum_wasm_bg.wasm from adjacent file
        """

    # Escape scene JSON for embedding in JS (handle </script> and special chars)
    escaped_scene = scene_json.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape_html(title)}</title>
  <style>
    body {{ margin: 0; padding: 20px; font-family: sans-serif; background: #fafafa; }}
    {css}
  </style>
</head>
<body>
  <div id="ferrum-chart"></div>

  <script type="module">
    // --- wasm-pack generated glue (inlined) ---
    {_inline_esm_as_blob(wasm_glue_js)}

    // --- ferrum interactive module (inlined) ---
    {_inline_module_code(interactive_js)}

    // --- SceneGraph data ---
    const SCENE_JSON = `{escaped_scene}`;

    // --- Initialize and render ---
    try {{
      {wasm_init_code}
      const el = document.getElementById('ferrum-chart');
      await renderChart({{
        el,
        sceneJson: SCENE_JSON,
        width: {width},
        height: {height},
      }});
    }} catch (e) {{
      console.error('ferrum render error:', e);
      document.getElementById('ferrum-chart').textContent =
        'Chart rendering failed: ' + e;
    }}
  </script>
</body>
</html>"""
    return html


def _escape_html(s: str) -> str:
    """Minimal HTML escaping for title text."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_esm_as_blob(js_source: str) -> str:
    """Convert wasm-pack ESM glue to inline script.

    wasm-pack generates ESM with `import.meta.url` and `fetch()` calls
    for loading the .wasm file. For inline embedding, we rewrite the
    init function to accept raw bytes instead.

    Returns JS code that defines `init` and `WasmRenderer` in the
    enclosing module scope.
    """
    # The wasm-pack glue exports an `init` function and the WasmRenderer class.
    # For inline use, we wrap it so `init(wasmBytes)` works with raw bytes.
    return f"""
    // wasm-pack glue (modified for inline WASM)
    let init, WasmRenderer;
    {{
      {js_source}
      init = __wbg_init;  // wasm-pack's default export
      WasmRenderer = _WasmRenderer || globalThis.WasmRenderer;
    }}
    """


def _inline_module_code(js_source: str) -> str:
    """Inline the ferrum-interactive.js module.

    Extracts the renderChart function for use in the script scope.
    """
    return f"""
    // ferrum-interactive.js (inlined)
    let renderChart;
    {{
      {js_source}
      // The module exports renderChart; capture it.
    }}
    """
```

**Design note on WASM inlining:** The wasm-pack-generated JS glue (`ferrum_wasm.js`) uses `import.meta.url` to locate the `.wasm` file via `fetch()`. For single-file HTML distribution, we base64-encode the WASM binary and pass the decoded bytes directly to the `init()` function. wasm-pack's `init(module_or_path)` accepts a `BufferSource` (Uint8Array), so `init(wasmBytes)` works. The exact inlining strategy in `_inline_esm_as_blob` will need adjustment based on the actual wasm-pack output — the function names vary by wasm-pack version.

- [ ] **Step 4: Create .gitkeep and update .gitignore**

Create `src/ferrum/_wasm/.gitkeep`:

```bash
mkdir -p src/ferrum/_wasm
touch src/ferrum/_wasm/.gitkeep
```

Add to `.gitignore`:

```
# WASM build artifacts (generated by wasm-pack, not committed)
src/ferrum/_wasm/*.wasm
src/ferrum/_wasm/*.js
src/ferrum/_wasm/*.d.ts
src/ferrum/_wasm/package.json
src/ferrum/_wasm/snippets/
!src/ferrum/_wasm/.gitkeep
!src/ferrum/_wasm/ferrum-interactive.js
!src/ferrum/_wasm/ferrum-interactive.css
```

**Note:** The hand-authored `ferrum-interactive.js` and `ferrum-interactive.css` are committed. The wasm-pack-generated files (`ferrum_wasm.js`, `ferrum_wasm_bg.wasm`, etc.) are gitignored — they are build artifacts.

- [ ] **Step 5: Verify HTML generation end-to-end**

Generate a test HTML file:

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum
unset CONDA_PREFIX && uv run --no-sync python -c "
import polars as pl
import ferrum as fm
from ferrum._core import render_interactive
from ferrum._html import build_html_bundle

# Generate scatter chart HTML
df = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 50.0, 30.0, 80.0, 60.0]})
chart = fm.Chart(df).mark_point().encode(x='x', y='y')
spec = chart._build_spec()
batch = chart._build_batch()
scene_json = render_interactive(spec, batch, viewport=(600.0, 400.0))
html = build_html_bundle(scene_json, width=600, height=400, title='Scatter Test')
with open('/tmp/ferrum_test_scatter.html', 'w') as f:
    f.write(html)
print(f'Written: /tmp/ferrum_test_scatter.html ({len(html)} bytes)')
"
```

**Manual verification:**
1. Open `/tmp/ferrum_test_scatter.html` in Chrome.
2. Checklist:
   - [ ] Five circles visible at distinct positions
   - [ ] Circles use the default theme mark color (steel blue)
   - [ ] Background is white
   - [ ] X-axis tick labels visible and positioned below the plot area
   - [ ] Y-axis tick labels visible and positioned left of the plot area
   - [ ] Axis titles visible ("x", "y")
   - [ ] No red errors in the browser dev console
   - [ ] Canvas fills the expected 600x400 viewport

- [ ] **Step 6: Commit**

```
feat(wasm): add JS glue module, CSS text overlay, and HTML bundle assembly
```

---

## Task 11b4: Python save API + packaging

Wire `save("chart.html")` and `save("chart.json")` into the existing `display.py` dispatcher, and configure maturin to include WASM artifacts in the wheel.

**Files:**
- Modify: `src/ferrum/display.py`
- Modify: `pyproject.toml`

### Steps

- [ ] **Step 1: Update display.py — wire html and json formats**

Replace the `NotImplementedError` branches in `save_chart`:

```python
def save_chart(
    chart: "Chart", path: Union[str, Path], *, format: str | None = None, **render_kwargs
) -> None:
    """Save a chart to disk as SVG, PNG, HTML, or JSON.

    The output format is derived from ``path``'s file extension when
    ``format`` is not supplied.

    Parameters
    ----------
    chart : Chart
        The chart to save.
    path : str or Path
        Destination file path.  The extension determines the format unless
        ``format`` is given explicitly.
    format : {"svg", "png", "html", "json"}, optional
        Explicit format override.  When omitted the extension of ``path``
        is used.  Raises ``ValueError`` if the path has no extension and
        ``format`` is also omitted.
    **render_kwargs
        Additional keyword arguments forwarded to the underlying render
        function.  For ``"html"`` format, accepted kwargs include
        ``embed_wasm`` (bool, default True) and ``title`` (str).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the format cannot be determined or is unsupported.
    FileNotFoundError
        If WASM artifacts are missing (for ``"html"`` format only).

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    >>> fm.save_chart(chart, "scatter.svg")
    >>> fm.save_chart(chart, "scatter.png")
    >>> fm.save_chart(chart, "scatter.html")
    >>> fm.save_chart(chart, "scatter.json")
    """
    path = Path(path)
    fmt = format or path.suffix.lstrip(".").lower()
    if fmt == "svg":
        path.write_text(chart.show_svg(**render_kwargs))
    elif fmt == "png":
        path.write_bytes(chart.show_png(**render_kwargs))
    elif fmt == "html":
        _save_html(chart, path, **render_kwargs)
    elif fmt == "json":
        _save_json(chart, path, **render_kwargs)
    elif fmt == "":
        raise ValueError(f"save({str(path)!r}) requires a format= or a path with extension.")
    else:
        raise ValueError(
            f"unknown extension {fmt!r}; supported: svg, png, html, json."
        )


def _save_html(chart: "Chart", path: Path, **kwargs) -> None:
    """Save chart as self-contained HTML with WASM renderer."""
    from ferrum._html import build_html_bundle

    # Get SceneGraph JSON from the Rust renderer
    scene_json = chart._render_scene_json(**{
        k: v for k, v in kwargs.items()
        if k not in ("embed_wasm", "title")
    })

    # Determine viewport from chart or defaults
    width = kwargs.get("width", 600.0)
    height = kwargs.get("height", 400.0)

    html = build_html_bundle(
        scene_json,
        width=width,
        height=height,
        title=kwargs.get("title", chart._title or "Ferrum chart"),
        embed_wasm=kwargs.get("embed_wasm", True),
    )
    path.write_text(html, encoding="utf-8")


def _save_json(chart: "Chart", path: Path, **kwargs) -> None:
    """Save chart as SceneGraph JSON."""
    scene_json = chart._render_scene_json(**kwargs)
    path.write_text(scene_json, encoding="utf-8")
```

- [ ] **Step 2: Add _render_scene_json method to Chart**

In `src/ferrum/chart.py`, add a method that calls `render_interactive`:

```python
def _render_scene_json(self, **render_kwargs) -> str:
    """Render the chart to SceneGraph JSON (internal API).

    Returns
    -------
    str
        JSON string containing the serialized SceneGraph IR.
        This is the internal IR, not a Vega-Lite spec.
    """
    from ferrum._core import render_interactive

    spec = self._build_spec()
    batch = self._build_batch()
    viewport = (
        render_kwargs.pop("width", 600.0),
        render_kwargs.pop("height", 400.0),
    )
    theme = self._build_theme_dict()
    return render_interactive(spec, batch, viewport=viewport, theme=theme)
```

This method follows the same pattern as the existing `show_svg()` and `show_png()` methods on Chart, calling through to the Rust binding. **Before implementing:** read the existing `show_svg()` method in `chart.py` to confirm the internal API names (`_build_spec()`, `_build_batch()`, `_build_theme_dict()`) and follow its exact pattern for spec/batch/theme construction and viewport handling. The method names shown above are the expected names based on the 11a implementation, but the implementer should verify against the actual source.

- [ ] **Step 3: Update pyproject.toml — include WASM artifacts in wheel**

Add to the `[tool.maturin]` section:

```toml
[tool.maturin]
module-name = "ferrum._core"
manifest-path = "crates/ferrum-core/Cargo.toml"
python-source = "src"
features = ["extension-module"]
strip = true
include = [
    "src/ferrum/_wasm/*.wasm",
    "src/ferrum/_wasm/*.js",
    "src/ferrum/_wasm/*.css",
]
```

This ensures `maturin develop` and `maturin build` include the WASM binary, the wasm-pack JS glue, and the hand-authored CSS/JS files in the wheel.

**Build order is load-bearing:**
1. `wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/` — produces WASM artifacts
2. `unset CONDA_PREFIX && uv run --no-sync maturin develop` — packages them into the wheel

If step 2 runs before step 1, the WASM files won't be in the wheel.

- [ ] **Step 4: Build and test the full pipeline**

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum

# Step 1: Build WASM
wasm-pack build crates/ferrum-wasm --target web --dev --out-dir ../../src/ferrum/_wasm/

# Step 2: Rebuild Python extension (picks up WASM artifacts)
unset CONDA_PREFIX && uv run --no-sync maturin develop

# Step 3: Test save("chart.html")
uv run python -c "
import polars as pl
import ferrum as fm

df = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 50.0, 30.0, 80.0, 60.0]})
chart = fm.Chart(df).mark_point().encode(x='x', y='y')
chart.save('/tmp/ferrum_scatter.html')
print('HTML saved: /tmp/ferrum_scatter.html')

chart.save('/tmp/ferrum_scatter.json')
print('JSON saved: /tmp/ferrum_scatter.json')
"

# Step 4: Test save("chart.json") produces valid JSON
uv run python -c "
import json
with open('/tmp/ferrum_scatter.json') as f:
    scene = json.load(f)
assert 'width' in scene
assert 'height' in scene
assert 'panels' in scene
print(f'Valid SceneGraph JSON: {len(scene[\"panels\"])} panels')
"
```

Expected:
- HTML file written, ~2-4 MB (most is the base64-encoded WASM binary).
- JSON file written, ~10-50 KB.
- JSON round-trips cleanly and contains the expected SceneGraph structure.

- [ ] **Step 5: Generate test HTML files for all golden fixture types**

Generate HTML files for the four golden test fixtures (scatter, bar, line, area) to verify visual correctness across mark types:

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum
uv run python -c "
import polars as pl
import ferrum as fm

# 1. Scatter
df = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 50.0, 30.0, 80.0, 60.0]})
fm.Chart(df).mark_point().encode(x='x', y='y').save('/tmp/ferrum_scatter.html')

# 2. Bar
df = pl.DataFrame({'cat': ['A', 'B', 'C', 'D'], 'val': [3.0, 1.0, 4.0, 1.5]})
fm.Chart(df).mark_bar().encode(x='cat:O', y='val').save('/tmp/ferrum_bar.html')

# 3. Line
df = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 50.0, 30.0, 80.0, 60.0]})
fm.Chart(df).mark_line().encode(x='x', y='y').save('/tmp/ferrum_line.html')

# 4. Area
df = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 50.0, 30.0, 80.0, 60.0]})
fm.Chart(df).mark_area().encode(x='x', y='y').save('/tmp/ferrum_area.html')

print('All 4 HTML files written to /tmp/')
"
```

**Manual verification checklist for each file:**

Open each in Chrome (or Firefox). For each chart:

| Check | scatter | bar | line | area |
|---|---|---|---|---|
| Marks visible at correct positions | [ ] | [ ] | [ ] | [ ] |
| Mark color matches theme default | [ ] | [ ] | [ ] | [ ] |
| Background is white | [ ] | [ ] | [ ] | [ ] |
| X-axis tick labels readable | [ ] | [ ] | [ ] | [ ] |
| Y-axis tick labels readable | [ ] | [ ] | [ ] | [ ] |
| Axis titles visible | [ ] | [ ] | [ ] | [ ] |
| No console errors | [ ] | [ ] | [ ] | [ ] |
| Canvas fills expected viewport | [ ] | [ ] | [ ] | [ ] |

- [ ] **Step 6: Verify existing tests still pass**

```bash
cd /Users/chrissantiago/Dropbox/GitHub/ferrum

# Rust tests
source ~/.cargo/env
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-scene -p ferrum-core

# Python tests
uv run pytest tests/ -v
```

Expected: all pass. No regressions — 11b adds new code and modifies only `display.py`.

- [ ] **Step 7: Commit**

```
feat(display): wire save("chart.html") and save("chart.json") via WASM renderer
```

---

## Validation checklist (run before marking 11b done)

### Rust / WASM

- [ ] `cargo build -p ferrum-wasm --target wasm32-unknown-unknown` — compiles without errors
- [ ] `wasm-pack build crates/ferrum-wasm --target web --dev --out-dir ../../src/ferrum/_wasm/` — produces WASM + JS artifacts
- [ ] `cargo test -p ferrum-scene` — SceneGraph serde round-trip
- [ ] `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core` — all existing Rust tests pass
- [ ] `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — no clippy warnings
- [ ] No `unwrap()` in `ferrum-wasm` (enforced by `#[deny(clippy::unwrap_used)]`)

### Python

- [ ] `uv run pytest tests/ -v` — all existing Python tests pass
- [ ] `save("chart.html")` produces a valid self-contained HTML file
- [ ] `save("chart.json")` produces valid SceneGraph JSON
- [ ] SceneGraph JSON round-trips: `json.loads(chart._render_scene_json())` succeeds
- [ ] `save("chart.html", embed_wasm=False)` works (references sidecar .wasm file)

### Manual browser verification

- [ ] Scatter chart: 5 circles at distinct positions, correct colors, axis labels visible
- [ ] Bar chart: 4 bars at correct heights, correct positions
- [ ] Line chart: polyline connecting 5 points, correct stroke
- [ ] Area chart: filled area below the line, correct opacity
- [ ] All 4 charts: no red errors in browser dev console
- [ ] All 4 charts: text overlay aligns with canvas geometry (axis labels match tick marks)

### Packaging

- [ ] `unset CONDA_PREFIX && uv run --no-sync maturin develop` includes WASM artifacts
- [ ] `python -c "from pathlib import Path; p = Path(import('ferrum').__file__).parent / '_wasm'; print(list(p.glob('*.wasm')))"` — WASM file present in installed package

### No regressions

- [ ] All golden SVGs byte-identical (11b does not modify ferrum-core rendering)
- [ ] `render_interactive` binding returns valid SceneGraph JSON (tested in 11a, verified still works)
- [ ] No new Python dependencies added (anywidget is 11c scope)

---

## Known limitations (documented, not bugs)

| Limitation | Status | Resolution phase |
|---|---|---|
| `SceneNode::Raw` nodes skipped in WASM (legend colorbar gradients) | `console.warn` | 11c/11d: typed gradient representation |
| `SceneNode::Image` not rendered in WASM | `console.warn` | 11b follow-up or 11d (mark_image) |
| Draw order simplified (mesh → rect → circle) | Correct for common charts | 11c: sort by batch index for interleaved types |
| No interactions (hover, click, zoom, pan) | Static rendering only | 11c: selections + zoom/pan + anywidget |
| No `anywidget` / Jupyter integration | Standalone HTML only | 11c: InteractiveChart widget |
| `_inline_esm_as_blob` may need adjustment for wasm-pack output format | Manual tuning expected | During 11b3 implementation |
| `TextBaseline::Custom(s)` passthrough to CSS may not match SVG semantics | Rare edge case | Audit after 11d when more text marks tested |
