use web_sys::HtmlCanvasElement;
use wgpu::{
    Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration,
    TextureFormat, TextureUsages,
};

use crate::error::WasmRenderError;

#[derive(Debug)]
struct WebDisplay;

impl raw_window_handle::HasDisplayHandle for WebDisplay {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::web())
    }
}

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub format: TextureFormat,
}

pub async fn init_gpu(canvas: HtmlCanvasElement) -> Result<GpuContext, WasmRenderError> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::BROWSER_WEBGPU | Backends::GL,
        display: Some(Box::new(WebDisplay)),
        ..InstanceDescriptor::new_without_display_handle()
    });

    let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    let surface = instance
        .create_surface(surface_target)
        .map_err(|e| WasmRenderError::GpuInit(format!("create_surface: {e}")))?;

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| WasmRenderError::GpuInit(format!("request_adapter: {e}")))?;

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("ferrum"),
            required_features: wgpu::Features::empty(),
            required_limits: Limits::downlevel_webgl2_defaults(),
            ..Default::default()
        })
        .await
        .map_err(|e| WasmRenderError::GpuInit(format!("request_device: {e}")))?;

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(caps.formats[0]);

    let width = canvas.width().max(1);
    let height = canvas.height().max(1);

    let alpha_mode = if caps
        .alpha_modes
        .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
    {
        wgpu::CompositeAlphaMode::PreMultiplied
    } else {
        caps.alpha_modes[0]
    };

    let config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode,
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    Ok(GpuContext {
        device,
        queue,
        surface,
        config,
        format,
    })
}
