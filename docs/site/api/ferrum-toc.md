# ferrum

Top-level public surface of the `ferrum` package.

| Module | Description |
|---|---|
| [Chart](chart.md) | The `Chart` class — declaration, encoding, marks, theming, rendering |
| [Marks](marks.md) | All `.mark_*()` methods on `Chart` |
| [Encoding](encoding.md) | Typed encoding channels — `X`, `Y`, `Color`, `Size`, `Tooltip`, etc. |
| [Plots](plots.md) | Figure-level helpers — `displot`, `catplot`, `lmplot`, `pairplot`, etc. |
| [Visualizers](visualizers.md) | sklearn-style `.fit()` / `.score()` / `.show()` diagnostic classes |
| [Model Sources](model_sources.md) | `ModelSource` / `ComparedModelSource` adapters that feed the diagnostics |
| [Themes](themes.md) | 12 built-in themes and `Theme` / `set_default_theme` / `theme_context` |
| [Selection](selection.md) | `selection_point`, `selection_interval`, conditional encodings |
| [Parameters](parameters.md) | Reactive parameters — `param`, `Parameter`, `VariableParameter` |
| [Composition](composition.md) | `HConcatChart`, `VConcatChart`, `LayerChart`, `hconcat`, `vconcat` |
| [Annotations](annotations.md) | `annotate_text`, `annotate_line`, `annotate_rect`, label helpers |
| [Structural](structural.md) | `BreakAxis`, `Inset`, `SecondaryY` view modifiers |
| [Transforms](transforms.md) | `transform_filter`, `transform_aggregate`, `transform_calculate`, etc. |
| [Statistics](statistics.md) | Statistical transform value-objects — `Kde`, `Smooth`, `Bin`, `Violin`, etc. |
| [Scales](scales.md) | `Scale`, `LinearScale`, `LogScale`, `OrdinalScale`, etc. |
| [Schemes](schemes.md) | Color scheme helpers |
| [Coord](coord.md) | Coordinate systems — flip, polar, theta, radial |
| [Position](position.md) | Position adjustments — dodge, stack, jitter, nudge |
| [Layer](layer.md) | `Layer` class for explicit layer construction |
| [Repeat](repeat.md) | `RepeatChart` for faceted repeat patterns |
| [Axis](axis.md) | `Axis` value class for axis configuration |
| [Grid](grid.md) | `Grid` value class for gridline configuration |
| [Legend](legend.md) | `Legend` value class for legend configuration |
| [Title](title.md) | `Title` value class for title configuration |
| [Render Config](render_config.md) | `RenderConfig` for auto-raster, scale, and output tuning |
| [Color](color.md) | `ferrum.color` — color utilities and palette access |
| [Config](config.md) | `ferrum.config` — runtime configuration |
