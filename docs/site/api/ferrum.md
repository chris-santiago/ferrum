# ferrum

Top-level public surface of the `ferrum` package. Everything below is importable directly from `import ferrum as fm`.

## Core

| Symbol | Description |
|--------|-------------|
| [`Chart`](chart.md) | The central chart object — binddata, set a mark, encode channels |
| [`Layer`](layer.md) | Overlay multiple marks on shared axes |

## Marks

54 mark methods on `Chart` — see the [Marks reference](marks.md) for the full list.

## Encoding channels

Channel constructors (`X`, `Y`, `Color`, `Size`, `Opacity`, `Shape`, `Tooltip`, etc.) for mapping data columns to visual properties.

See [ferrum.encoding](encoding.md).

## Scales

`LinearScale`, `LogScale`, `PowScale`, `SqrtScale`, `OrdinalScale`, `BandScale`, `PointScale`, `TimeScale`, `SequentialScale`, `DivergingScale`, `QuantizeScale`, `BinOrdinalScale`.

See [ferrum.scales](scales.md).

## Transforms

`transform_filter`, `transform_calculate`, `transform_aggregate`, `transform_bin`, `transform_kde`, `transform_window`, `transform_fold`, `transform_pivot`, `transform_flatten`, `transform_sample`, `transform_quantile`, `transform_loess`, `transform_regression`, `transform_contour`, `transform_reorder`, `transform_impute`, `transform_joinaggregate`.

See [ferrum.transforms](transforms.md).

## Composition

Operators for combining charts into compound views.

| Symbol | Description |
|--------|-------------|
| [`hconcat`](composition.md) | Horizontal concatenation |
| [`vconcat`](composition.md) | Vertical concatenation |
| [`FacetChart`](composition.md) | Facet by a column |
| [`RepeatChart`](repeat.md) | Repeat a template across fields |
| [`JointChart`](composition.md) | Joint plot with marginals |

## Coordinate systems

`CoordCartesian`, `CoordPolar`, `CoordGeo` — see [ferrum.coord](coord.md).

## Position adjustments

`Dodge`, `Stack`, `Jitter`, `Nudge` — see [ferrum.position](position.md).

## Selections & interactivity

`selection_point`, `selection_interval`, `value`, `ConditionalSpec` — see [ferrum.selection](selection.md).

## Themes

12 built-in themes and the `Theme` constructor — see [ferrum.themes](themes.md).

## Figure helpers (plots)

44 high-level figure functions (`roc_chart`, `confusion_matrix_chart`, `residuals_chart`, `shap_summary_chart`, etc.) — see [ferrum.plots](plots.md).

## Visualizers

28 sklearn-protocol diagnostic visualizers (`ROCVisualizer`, `CalibrationVisualizer`, `SilhouetteVisualizer`, etc.) — see [Visualizers](visualizers.md).

## Annotations

`HLine`, `VLine`, `HBand`, `VBand`, `TextAnnotation` — see [ferrum.annotations](annotations.md).

## Color & schemes

`Gradient`, `continuous_palette` — see [ferrum.schemes](schemes.md) and [ferrum.color](color.md).

## Configuration

| Symbol | Description |
|--------|-------------|
| [`Axis`](axis.md) | Axis appearance and tick configuration |
| [`Legend`](legend.md) | Legend appearance and layout |
| [`Title`](title.md) | Chart title configuration |
| [`RenderConfig`](render_config.md) | Output size, DPI, format settings |
| [`ChartConfig`](config.md) | Global chart configuration |

::: ferrum
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
