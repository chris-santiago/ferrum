use web_sys::HtmlCanvasElement;
use wgpu::{
    Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration,
    TextureFormat, TextureUsages,
};

use crate::error::WasmRenderError;
use crate::msaa::choose_sample_count;

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
    /// MSAA sample count shared by the main render pass and every pipeline in
    /// it. `4` when the backend reports 4× support, else `1` (FA-19 fallback —
    /// `sample_count == 1` is byte-identical to the pre-MSAA render path).
    pub sample_count: u32,
    /// Multisampled color target the main pass renders into, resolving to the
    /// single-sample surface view. `None` when `sample_count == 1`.
    pub msaa_view: Option<wgpu::TextureView>,
}

/// Build the multisampled color target for the main pass, or `None` at 1×.
///
/// Sized to the current surface config; recreated on construction and on every
/// resize so the MSAA target always matches the surface dimensions.
fn build_msaa_view(
    device: &Device,
    config: &SurfaceConfiguration,
    format: TextureFormat,
    sample_count: u32,
) -> Option<wgpu::TextureView> {
    if sample_count <= 1 {
        return None;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

impl GpuContext {
    /// Recreate the MSAA color target from the current `config` (call after a
    /// resize so the target matches the new surface dimensions).
    pub fn rebuild_msaa_view(&mut self) {
        self.msaa_view =
            build_msaa_view(&self.device, &self.config, self.format, self.sample_count);
    }
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

    // FA-19: enable 4× MSAA on the main pass when the backend supports it for
    // the chosen surface format; otherwise fall back to 1× (no MSAA).
    let supports_4x = adapter
        .get_texture_format_features(format)
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4);
    let sample_count = choose_sample_count(supports_4x);

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

    let msaa_view = build_msaa_view(&device, &config, format, sample_count);

    Ok(GpuContext {
        device,
        queue,
        surface,
        config,
        format,
        sample_count,
        msaa_view,
    })
}
