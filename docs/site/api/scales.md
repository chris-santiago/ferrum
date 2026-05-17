# Scales

Scale objects map data values to visual properties (pixel positions, colors, sizes).
Pass a scale instance to an encoding channel's `scale=` parameter for explicit control
over the mapping, or omit it to let ferrum infer scale type and domain from the data.

All scale classes are importable from the top-level `ferrum` namespace:

```python
import ferrum as fm

chart = (
    fm.Chart(df)
    .mark_point()
    .encode(
        x=fm.X("price", scale=fm.LogScale(domain=[1, 10000], range=[0, 600], base=10)),
        y="quantity",
    )
)
```

## Continuous scales

Continuous scales map a numeric domain to a numeric range.

### LinearScale

```python
fm.LinearScale(*, domain, range, clamp=False, nice=False, padding=None)
```

Linear (identity) mapping from domain to range.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input domain `[min, max]`. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `clamp` | `bool` | `False` | Clamp output to range bounds. |
| `nice` | `bool` | `False` | Extend domain to nice round numbers. |
| `padding` | `float \| None` | `None` | Padding added to both ends of the domain. |

**Methods:** `scale(x)`, `invert(y)`, `ticks(count=10)`, `nice()`

### LogScale

```python
fm.LogScale(*, domain, range, base=10.0, clamp=False, nice=False, padding=None)
```

Logarithmic mapping. Domain must not include zero.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input domain `[min, max]`. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `base` | `float` | `10.0` | Logarithm base. |
| `clamp` | `bool` | `False` | Clamp output to range bounds. |
| `nice` | `bool` | `False` | Extend domain to nice round numbers. |
| `padding` | `float \| None` | `None` | Padding added to both ends of the domain. |

**Methods:** `scale(x)`, `invert(y)`, `ticks(count=10)`, `nice()`

### PowScale

```python
fm.PowScale(*, domain, range, exponent=2.0, clamp=False, nice=False, padding=None)
```

Power (polynomial) mapping: `y = x^exponent`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input domain `[min, max]`. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `exponent` | `float` | `2.0` | Power exponent. |
| `clamp` | `bool` | `False` | Clamp output to range bounds. |
| `nice` | `bool` | `False` | Extend domain to nice round numbers. |
| `padding` | `float \| None` | `None` | Padding added to both ends of the domain. |

**Methods:** `scale(x)`, `invert(y)`, `ticks(count=10)`, `nice()`

### SqrtScale

```python
fm.SqrtScale(*, domain, range, clamp=False, nice=False, padding=None)
```

Square-root mapping (equivalent to `PowScale` with `exponent=0.5`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input domain `[min, max]`. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `clamp` | `bool` | `False` | Clamp output to range bounds. |
| `nice` | `bool` | `False` | Extend domain to nice round numbers. |
| `padding` | `float \| None` | `None` | Padding added to both ends of the domain. |

**Methods:** `scale(x)`, `invert(y)`, `ticks(count=10)`, `nice()`

### SymlogScale

```python
fm.SymlogScale(*, domain, range, constant=1.0, clamp=False, nice=False, padding=None)
```

Symmetric log mapping that handles zero and negative values.
Uses `sign(x) * log1p(|x| / constant)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input domain `[min, max]`. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `constant` | `float` | `1.0` | Symlog constant (linearity threshold). |
| `clamp` | `bool` | `False` | Clamp output to range bounds. |
| `nice` | `bool` | `False` | Extend domain to nice round numbers. |
| `padding` | `float \| None` | `None` | Padding added to both ends of the domain. |

**Methods:** `scale(x)`, `invert(y)`, `ticks(count=10)`, `nice()`

### TimeScale

```python
fm.TimeScale(*, domain, range, clamp=False, nice=False, padding=None)
```

Temporal scale for datetime values. Domain values are epoch milliseconds.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input domain as epoch-ms `[start, end]`. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `clamp` | `bool` | `False` | Clamp output to range bounds. |
| `nice` | `bool` | `False` | Extend domain to nice temporal boundaries. |
| `padding` | `float \| None` | `None` | Padding added to both ends of the domain. |

**Methods:** `scale(x)`, `invert(y)`, `ticks(count=10)`, `nice()`

## Ordinal scales

Ordinal scales map discrete categories to positions or visual values.

### OrdinalScale

```python
fm.OrdinalScale(*, domain, range, padding=0.0)
```

Maps discrete string categories to evenly-spaced numeric positions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[str]` | required | Ordered list of category names. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `padding` | `float` | `0.0` | Outer padding as a fraction of step size. |

**Methods:** `scale(value)`, `invert(y)`, `ticks()`

### BandScale

```python
fm.BandScale(*, domain, range, padding_inner=0.1, padding_outer=0.05, align=0.5)
```

Maps discrete categories to bands (rectangles) with configurable inner and outer padding.
Used internally for bar charts.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[str]` | required | Ordered list of category names. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `padding_inner` | `float` | `0.1` | Space between bands as a fraction of step. |
| `padding_outer` | `float` | `0.05` | Space before first and after last band. |
| `align` | `float` | `0.5` | Band alignment within step (0 = left, 1 = right). |

### PointScale

```python
fm.PointScale(*, domain, range, padding=0.5, align=0.5, reverse=False)
```

Maps discrete categories to individual points (zero-width bands).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[str]` | required | Ordered list of category names. |
| `range` | `Sequence[float]` | required | Output range `[min, max]`. |
| `padding` | `float` | `0.5` | Outer padding as a fraction of step. |
| `align` | `float` | `0.5` | Point alignment within step. |
| `reverse` | `bool` | `False` | Reverse the scale direction. |

## Color scales

Color scales map numeric data to color ramps for continuous color encoding.

### SequentialScale

```python
fm.SequentialScale(*, scheme=None, domain=None, reverse=False)
```

Maps a continuous domain to a sequential color scheme (light-to-dark).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scheme` | `str \| None` | `None` | Named color scheme (e.g. `"viridis"`, `"blues"`). |
| `domain` | `Sequence[float] \| None` | `None` | Input domain `[min, max]`. |
| `reverse` | `bool` | `False` | Reverse the color ramp direction. |

### DivergingScale

```python
fm.DivergingScale(*, scheme=None, domain=None, domain_mid=None)
```

Maps a continuous domain to a diverging color scheme (two-hue ramp with a neutral midpoint).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scheme` | `str \| None` | `None` | Named diverging scheme (e.g. `"rdbu"`, `"blue_to_red"`). |
| `domain` | `Sequence[float] \| None` | `None` | Input domain `[min, max]`. |
| `domain_mid` | `float \| None` | `None` | Midpoint value (defaults to domain center). |

### QuantizeScale

```python
fm.QuantizeScale(*, domain=None, range=None)
```

Quantizes a continuous domain into discrete bins mapped to a color range.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float] \| None` | `None` | Input domain `[min, max]`. |
| `range` | `Sequence[float] \| None` | `None` | Output values (one per bin). |

**Methods:** `scale(x)`, `invert_extent(y)`, `ticks()`

### BinOrdinalScale

```python
fm.BinOrdinalScale(*, bins=None, scheme=None)
```

Maps pre-binned data to an ordinal color scheme — each bin gets a distinct color.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bins` | `Sequence[float] \| None` | `None` | Bin boundaries. |
| `scheme` | `str \| None` | `None` | Named color scheme for the bins. |

## Classification scales

Classification scales divide a continuous domain into discrete segments.

### ThresholdScale

```python
fm.ThresholdScale(*, domain, range)
```

Maps values to range segments based on explicit threshold breakpoints.
`len(range)` should be `len(domain) + 1`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Threshold breakpoints. |
| `range` | `Sequence[float]` | required | Output values (one per segment). |

**Methods:** `scale(x)`, `invert_extent(y)`, `ticks()`

### QuantileScale

```python
fm.QuantileScale(*, domain, range)
```

Divides the sorted domain into quantile-based segments mapped to range values.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `domain` | `Sequence[float]` | required | Input data values (used to compute quantiles). |
| `range` | `Sequence[float]` | required | Output values (one per quantile segment). |

**Methods:** `scale(x)`, `invert_extent(y)`, `ticks(count=None)`
