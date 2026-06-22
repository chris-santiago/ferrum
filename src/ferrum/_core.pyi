from typing import Any, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

DataTypeStr = Literal[
    "Q",
    "N",
    "O",
    "T",
    "quantitative",
    "nominal",
    "ordinal",
    "temporal",
]
MarkStr = Literal[
    "point",
    "line",
    "bar",
    "area",
    "rule",
    "text",
    "tick",
    "rect",
    "polygon",
    "image",
    "ribbon",
]

def process_batch(reader: Any) -> Any:
    """
    Convert a Python record batch through the ferrum transform pipeline.

    Examples
    --------
    >>> import ferrum as fm
    >>> result = fm.process_batch(record_batch)
    """
    ...

class EncodingSpec:
    field: str
    type_: Optional[str]
    scale: Optional[Any]
    title: Optional[str]
    axis: Optional[Any]
    legend: Optional[Any]
    sort: Optional[Any]
    stack: Optional[str]
    condition: Optional[Any]
    impute: Optional[Any]
    scheme: Optional[str]
    format: Optional[str]
    format_type: Optional[str]
    def __init__(
        self,
        field: str,
        type_: Optional[DataTypeStr] = None,
        *,
        scale: Optional[Any] = None,
        title: Optional[str] = None,
        axis: Optional[Any] = None,
        legend: Optional[Any] = None,
        sort: Optional[Any] = None,
        stack: Optional[str] = None,
        condition: Optional[Any] = None,
        impute: Optional[Any] = None,
        scheme: Optional[str] = None,
        format: Optional[str] = None,
        format_type: Optional[str] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class ChartSpec:
    mark: str
    x: Optional[EncodingSpec]
    y: Optional[EncodingSpec]
    color: Optional[EncodingSpec]
    size: Optional[EncodingSpec]
    shape: Optional[EncodingSpec]
    opacity: Optional[EncodingSpec]
    x2: Optional[EncodingSpec]
    y2: Optional[EncodingSpec]
    data: str
    transforms: List[object]
    facet: Optional[dict]
    layers: Optional[List[dict]]
    coord: Optional[str]
    mark_style: Optional[dict]

    def __init__(
        self,
        *,
        mark: MarkStr,
        x: Union[str, EncodingSpec, None] = None,
        y: Union[str, EncodingSpec, None] = None,
        color: Union[str, EncodingSpec, None] = None,
        size: Union[str, EncodingSpec, None] = None,
        shape: Union[str, EncodingSpec, None] = None,
        opacity: Union[str, EncodingSpec, None] = None,
        x2: Union[str, EncodingSpec, None] = None,
        y2: Union[str, EncodingSpec, None] = None,
        text: Union[str, EncodingSpec, None] = None,
        tooltip: Union[str, EncodingSpec, None] = None,
        href: Union[str, EncodingSpec, None] = None,
        description: Union[str, EncodingSpec, None] = None,
        tooltip_fields: Optional[str] = None,
        data: Optional[str] = None,
        transforms: Optional[List[object]] = None,
        transforms_json: Optional[str] = None,
        layers: Optional[List[object]] = None,
        coord: Optional[Any] = None,
        facet: Optional[Any] = None,
        mark_style: Optional[Any] = None,
        position: Optional[Any] = None,
        title: Optional[Any] = None,
        axis_x: Optional[bool] = None,
        axis_y: Optional[bool] = None,
        key: Union[str, EncodingSpec, None] = None,
        selections: Optional[str] = None,
        conditionals: Optional[str] = None,
        params: Optional[str] = None,
        url: Union[str, EncodingSpec, None] = None,
        stroke_opacity: Union[str, EncodingSpec, None] = None,
        stroke_width: Union[str, EncodingSpec, None] = None,
        stroke_dash: Union[str, EncodingSpec, None] = None,
        angle: Union[str, EncodingSpec, None] = None,
        fill_opacity: Union[str, EncodingSpec, None] = None,
        chart_description: Optional[str] = None,
    ) -> None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "ChartSpec": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

# ---------- Scales (Phase 4) ----------

class LinearScale:
    domain: list[float] | None
    range: list[float] | None
    clamp: bool
    padding: float | None
    def __init__(
        self,
        *,
        domain: Sequence[float] | None = None,
        range: Sequence[float] | None = None,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LinearScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class LogScale:
    domain: list[float] | None
    range: list[float] | None
    base: float
    clamp: bool
    padding: float | None
    def __init__(
        self,
        *,
        domain: Sequence[float] | None = None,
        range: Sequence[float] | None = None,
        base: float = 10.0,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class TimeScale:
    domain: list[float]
    range: list[float] | None
    clamp: bool
    padding: float | None
    utc: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float] | None = None,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
        utc: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "TimeScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class SymlogScale:
    domain: list[float] | None
    range: list[float] | None
    constant: float
    clamp: bool
    padding: float | None
    def __init__(
        self,
        *,
        domain: Sequence[float] | None = None,
        range: Sequence[float] | None = None,
        constant: float = 1.0,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "SymlogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class OrdinalScale:
    domain: list[str]
    range: list[float | str] | None
    padding: float
    def __init__(
        self,
        *,
        domain: Sequence[str],
        range: Sequence[float | str] | None = None,
        padding: float = 0.0,
    ) -> None: ...
    def scale(self, value: str) -> float: ...
    def invert(self, y: float) -> Optional[str]: ...
    def ticks(self) -> list[str]: ...
    def nice(self) -> "OrdinalScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class ThresholdScale:
    domain: list[float]
    range: list[float]
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert_extent(self, y: float) -> tuple[float, float]: ...
    def ticks(self) -> list[float]: ...
    def nice(self) -> "ThresholdScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class QuantileScale:
    domain: list[float]
    range: list[float]
    quantiles: list[float]
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert_extent(self, y: float) -> tuple[float, float]: ...
    def ticks(self, count: Optional[int] = None) -> list[float]: ...
    def nice(self) -> "QuantileScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class PowScale:
    domain: Optional[list[float]]
    range: Optional[list[float]]
    exponent: float
    clamp: bool
    padding: Optional[float]
    def __init__(
        self,
        *,
        domain: Optional[Sequence[float]] = None,
        range: Optional[Sequence[float]] = None,
        exponent: float = 2.0,
        clamp: bool = False,
        nice: bool = False,
        padding: Optional[float] = None,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "PowScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class SqrtScale:
    domain: Optional[list[float]]
    range: Optional[list[float]]
    exponent: float
    clamp: bool
    padding: Optional[float]
    def __init__(
        self,
        *,
        domain: Optional[Sequence[float]] = None,
        range: Optional[Sequence[float]] = None,
        clamp: bool = False,
        nice: bool = False,
        padding: Optional[float] = None,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "SqrtScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class BandScale:
    domain: list[str]
    range: Optional[list[float]]
    padding_inner: float
    padding_outer: float
    align: float
    def __init__(
        self,
        *,
        domain: Optional[Sequence[str]] = None,
        padding: float = 0.1,
        padding_inner: Optional[float] = None,
        padding_outer: Optional[float] = None,
        align: float = 0.5,
        range: Optional[Sequence[float]] = None,
    ) -> None: ...
    def scale(self, value: str) -> float: ...
    def bandwidth(self) -> float: ...
    def ticks(self) -> list[str]: ...
    def nice(self) -> "BandScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class PointScale:
    domain: list[str]
    range: Optional[list[float]]
    padding: float
    align: float
    reverse: bool
    def __init__(
        self,
        *,
        domain: Optional[Sequence[str]] = None,
        padding: float = 0.5,
        align: float = 0.5,
        reverse: bool = False,
        range: Optional[Sequence[float]] = None,
    ) -> None: ...
    def scale(self, value: str) -> float: ...
    def ticks(self) -> list[str]: ...
    def nice(self) -> "PointScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class SequentialScale:
    scheme: Optional[str]
    domain: Optional[list[float]]
    reverse: bool
    def __init__(
        self,
        *,
        scheme: Optional[str] = None,
        domain: Optional[Sequence[float]] = None,
        reverse: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class DivergingScale:
    scheme: Optional[str]
    domain: Optional[list[float]]
    domain_mid: Optional[float]
    def __init__(
        self,
        *,
        scheme: Optional[str] = None,
        domain: Optional[Sequence[float]] = None,
        domain_mid: Optional[float] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class QuantizeScale:
    domain: list[float]
    range: list[str]
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[str],
    ) -> None: ...
    def scale(self, x: float) -> Optional[str]: ...
    def thresholds(self) -> list[float]: ...
    def nice(self) -> "QuantizeScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class BinOrdinalScale:
    bins: list[float]
    scheme: Optional[str]
    num_bins: int
    def __init__(
        self,
        *,
        bins: Sequence[float],
        scheme: Optional[str] = None,
    ) -> None: ...
    def scale(self, x: float) -> Optional[int]: ...
    def ticks(self) -> list[float]: ...
    def nice(self) -> "BinOrdinalScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

# ---------- Stat engine transforms (Phase 5) ----------

class Bin:
    def __init__(
        self,
        field: str,
        *,
        bin_count: Optional[int] = None,
        bin_width: Optional[float] = None,
        extent: Optional[Tuple[float, float]] = None,
        nice: bool = True,
        cumulative: bool = False,
        shared_extent: bool = False,
        groupby: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Bin2D:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        bins_x: Optional[Union[str, int, float]] = None,
        bins_y: Optional[Union[str, int, float]] = None,
        extent_x: Optional[Tuple[float, float]] = None,
        extent_y: Optional[Tuple[float, float]] = None,
        cumulative: bool = False,
        name: Optional[str] = None,
    ) -> None: ...

class BoxStats:
    def __init__(
        self,
        field: str,
        *,
        groupby: List[str] = ...,
        whisker_extent: Union[str, float] = 1.5,
        name: Optional[str] = None,
    ) -> None: ...

class Kde:
    def __init__(
        self,
        field: str,
        *,
        bandwidth: object = None,  # str ("scott"|"silverman") or float; None → scott
        bw_adjust: float = 1.0,
        n: int = 512,
        extent: Optional[Tuple[float, float]] = None,
        cumulative: bool = False,
        shared_extent: bool = False,
        kernel: str = "gaussian",
        groupby: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Kde2D:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        bandwidth: object = None,  # str ("scott"|"silverman") or float; None → scott
        n: int = 128,
        extent: Optional[Tuple[float, float, float, float]] = None,
        groupby: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Contour:
    def __init__(
        self,
        *,
        thresholds: int = 6,
        fill: bool = False,
        smooth: bool = True,
        name: Optional[str] = None,
    ) -> None: ...

class Smooth:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        method: str = "loess",
        ci: Optional[float] = 0.95,
        bandwidth: float = 0.75,
        degree: int = 2,
        n: int = 200,
        seed: int = 0,
        x_bins: Optional[int] = None,
        x_estimator: Optional[str] = None,
        output: str = "fitted",
        inject_zero_ref: bool = False,
        inject_metrics: bool = False,
        groupby: Optional[str] = None,
        name: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
    ) -> None: ...

class AggregateOp:
    def __init__(self, field: str, fn_: str, as_: str) -> None: ...

class Aggregate:
    def __init__(
        self,
        ops: List[AggregateOp],
        *,
        groupby: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Summary:
    def __init__(
        self,
        field: str,
        *,
        groupby: Optional[List[str]] = None,
        error_fn: str = "ci",
        ci: float = 0.95,
        n_boot: int = 1000,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None: ...

class Outliers:
    def __init__(
        self,
        field: str,
        *,
        groupby: List[str] = ...,
        extent: float = 1.5,
        name: Optional[str] = None,
    ) -> None: ...

class ErrorExtent:
    def __init__(
        self,
        field: str,
        *,
        method: str = "ci",
        groupby: List[str] = ...,
        seed: int = 0,
        n_boot: int = 1000,
        name: Optional[str] = None,
    ) -> None: ...

class Violin:
    def __init__(
        self,
        field: str,
        *,
        groupby: List[str] = ...,
        bandwidth: object = None,  # str ("scott"|"silverman") or float; None → scott
        bw_adjust: float = 1.0,
        n: int = 256,
        width: float = 0.4,
        shared_extent: bool = False,
        horizontal: bool = False,
        name: Optional[str] = None,
    ) -> None: ...

class QQ:
    def __init__(
        self,
        field: str,
        *,
        distribution: str = "normal",
        dequantize: bool = False,
        emit_line: bool = True,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None: ...

class Raster:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        aggregate: str = "count",
        field: Optional[str] = None,
        resolution: Optional[Union[str, int, Tuple[int, int]]] = None,
        min_count: Optional[int] = None,
        log_scale: bool = False,
        name: Optional[str] = None,
    ) -> None: ...

class Hex:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        bin_size: Optional[float] = None,
        aggregate: str = "count",
        field: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Swarm:
    def __init__(
        self,
        category: str,
        value: str,
        *,
        point_size: float = 5.0,
        spacing: float = 1.0,
        side: str = "both",
        orient: str = "vertical",
        dodge: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Linkage:
    def __init__(
        self,
        *,
        method: str = "ward",
        metric: str = "euclidean",
        axis: str = "rows",
        z_score: str | None = None,
        standard_scale: str | None = None,
        name: str | None = None,
    ) -> None: ...

class Reorder:
    def __init__(
        self,
        by: str,
        *,
        drop_index: bool = True,
        from_: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Unpivot:
    def __init__(
        self,
        *,
        id_vars: list[str] = ...,
        value_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
        name: str | None = None,
    ) -> None: ...

class LetterValue:
    def __init__(
        self,
        value: str,
        *,
        group: Optional[str] = None,
        k_depth: str = "proportion",
        k_proportion: float = 0.007,
        outlier_threshold: float = 1.5,
        name: Optional[str] = None,
        sort: Optional[dict[str, str]] = None,
    ) -> None: ...

class Logistic:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        n_grid: int = 100,
        ci: Optional[float] = None,
        max_iter: int = 25,
        tol: float = 1e-8,
        name: Optional[str] = None,
    ) -> None: ...

class Glm:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        family: str = "gaussian",
        link: Optional[str] = None,
        n_grid: int = 100,
        ci: Optional[float] = None,
        max_iter: int = 25,
        tol: float = 1e-8,
        name: Optional[str] = None,
    ) -> None: ...

class Robust:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        n_grid: int = 100,
        ci: Optional[float] = None,
        huber_c: float = 1.345,
        max_iter: int = 25,
        tol: float = 1e-8,
        output: str = "fitted",
        inject_zero_ref: bool = False,
        inject_metrics: bool = False,
        name: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
    ) -> None: ...

# ---------- Data transforms (Vega-lite-style, declared missing from stub) ----------

class DataAggregate:
    def __init__(
        self,
        *,
        groupby: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> None: ...

class DataBin:
    def __init__(
        self,
        field: str,
        *,
        as_: Optional[str] = None,
        maxbins: Optional[int] = None,
        step: Optional[float] = None,
        nice: bool = True,
        name: Optional[str] = None,
    ) -> None: ...

class DataCalculate:
    def __init__(
        self,
        expr: str,
        as_field: str,
        *,
        name: Optional[str] = None,
    ) -> None: ...

class DataFilter:
    def __init__(
        self,
        predicate: str,
        *,
        name: Optional[str] = None,
    ) -> None: ...

class DataFold:
    def __init__(
        self,
        fields: List[str],
        *,
        as_key: str = "key",
        as_value: str = "value",
        name: Optional[str] = None,
    ) -> None: ...

class DataPivot:
    def __init__(
        self,
        field: str,
        value: str,
        *,
        groupby: Optional[List[str]] = None,
        limit: Optional[int] = None,
        op: str = "sum",
        name: Optional[str] = None,
    ) -> None: ...

class DataStack:
    def __init__(
        self,
        field: str,
        groupby: List[str],
        *,
        offset: str = "zero",
        name: Optional[str] = None,
    ) -> None: ...

class DataWindow:
    def __init__(
        self,
        *,
        name: Optional[str] = None,
    ) -> None: ...

class DensityData:
    def __init__(
        self,
        field: str,
        *,
        name: Optional[str] = None,
    ) -> None: ...

class Flatten:
    def __init__(
        self,
        fields: List[str],
        *,
        as_: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> None: ...

class Impute:
    def __init__(
        self,
        field: str,
        *,
        method: str = "value",
        value: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None: ...

class JoinAggregate:
    def __init__(
        self,
        *,
        name: Optional[str] = None,
    ) -> None: ...

class LoessData:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        bandwidth: float = 0.3,
        name: Optional[str] = None,
    ) -> None: ...

class PyIdentity:
    def __init__(self, name: str) -> None: ...

class ReferenceLine:
    def __init__(
        self,
        x_field: str,
        y_field: str,
        *,
        x: Tuple[float, float] = (0.0, 1.0),
        y: Tuple[float, float] = (0.0, 1.0),
        name: Optional[str] = None,
    ) -> None: ...

class RegressionData:
    def __init__(
        self,
        x: str,
        y: str,
        *,
        method: str = "linear",
        order: int = 1,
        name: Optional[str] = None,
    ) -> None: ...

class Sample:
    def __init__(
        self,
        n: int,
        *,
        seed: int = 42,
        name: Optional[str] = None,
    ) -> None: ...

class TimeUnit:
    def __init__(
        self,
        field: str,
        unit: str,
        *,
        utc: bool = False,
        as_: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class TopK:
    def __init__(
        self,
        n: int,
        field: str,
        *,
        op: str = "sum",
        sort: str = "descending",
        name: Optional[str] = None,
    ) -> None: ...

def compute_layout(
    spec,
    *,
    viewport: tuple[float, float],
    x_tick_labels: list[str],
    y_tick_labels: list[str],
    x_title: str | None = None,
    y_title: str | None = None,
    facet_groups: list[tuple[str, str, int]] | None = None,
    legend_entries: list[tuple[str, str]] | None = None,
    legend_orient: str = "right",
    label_angle: float | None = None,
) -> dict:
    """
    Compute the layout plan (panel sizes, margins) for a chart spec.

    Examples
    --------
    >>> import ferrum as fm
    >>> layout = fm.compute_layout(chart_spec_json, viewport_json)
    """
    ...

def render_svg(
    spec: ChartSpec,
    data: Any,
    *,
    viewport: tuple[float, float],
    theme: Optional[dict] = None,
    config: Optional[dict] = None,
    chart_config: Optional[dict] = None,
) -> str:
    """
    Render a chart spec to an SVG string.

    Examples
    --------
    >>> import ferrum as fm
    >>> svg = fm.render_svg(chart_spec_json, layout_json, data_batch)
    """
    ...

def render_png(
    spec: ChartSpec,
    data: Any,
    *,
    viewport: tuple[float, float],
    theme: Optional[dict] = None,
    config: Optional[dict] = None,
    chart_config: Optional[dict] = None,
) -> bytes:
    """
    Render a chart spec to PNG bytes.

    Examples
    --------
    >>> import ferrum as fm
    >>> png_bytes = fm.render_png(chart_spec_json, layout_json, data_batch)
    """
    ...

# ---------- Theme key manifest (single source of truth) ----------
# The complete valid theme-key set and its color-typed subset, published from
# the Rust `ThemeOverridesSpec`. The Python `Theme` derives `_KNOWN_KEYS` from
# `theme_known_keys()` so the two contracts cannot drift.

def theme_known_keys() -> list[str]: ...
def theme_color_keys() -> list[str]: ...

# ---------- Palette registry accessors ----------
#
# Read-only views over the Rust palette registry (categorical + continuous).
# ``ferrum.color`` consumes these as the single source of truth instead of
# hand-mirroring hex tables.

def list_palettes() -> list[str]: ...
def palette_kind(name: str) -> Optional[str]: ...
def palette_colors(name: str) -> Optional[list[str]]: ...
def palette_sample(name: str, t: float) -> Optional[str]: ...

# ---------- Missing functions (previously undeclared) ----------

def calinski_harabasz_score(x_table: Any, labels: Any) -> float: ...
def figure_title_nodes(
    width: float,
    height: float,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    left_inset: Optional[float] = None,
    right_inset: Optional[float] = None,
    anchor: Optional[str] = None,
) -> Tuple[str, float, float]: ...
def mds_classical(
    x_table: Any,
    n_components: int,
    input_type: str = "features",
    metric: str = "euclidean",
) -> Any: ...
def pca_scores(x_table: Any, n_components: Optional[int] = None) -> Any: ...
def pca_variance(x_table: Any, n_components: Optional[int] = None) -> Any: ...
def py_rank1d(x_table: Any, algorithm: Any, top_k: Any) -> Any: ...
def py_rank1d_with_y(x_table: Any, y: Any, algorithm: Any, top_k: Any) -> Any: ...
def py_rank2d(x_table: Any, algorithm: Any) -> Any: ...
def py_rankdata_average(x: Any) -> Any: ...
def py_shapiro_w(x: Any) -> Any: ...
def rasterize_svg(svg: str, *, scale: float = 2.0) -> bytes: ...
def render_interactive(
    spec: Any,
    data: Any,
    *,
    viewport: Any,
    theme: Any = None,
    config: Any = None,
    chart_config: Any = None,
) -> Tuple[str, bytes]: ...
def silhouette_samples(x_table: Any, labels: Any, metric: str = "euclidean") -> Any: ...
def silhouette_score(x_table: Any, labels: Any, metric: str = "euclidean") -> float: ...
def tsne_embedding(
    x_table: Any,
    n_components: int,
    seed: int,
    perplexity: Optional[float] = None,
    learning_rate: Optional[float] = None,
    n_iter: Optional[int] = None,
) -> Any: ...
def umap_embedding(
    x_table: Any,
    n_components: int,
    seed: int,
    n_neighbors: Optional[int] = None,
    min_dist: Optional[float] = None,
    n_epochs: Optional[int] = None,
) -> Any: ...

# ---------- SVG compositor (Phase 8a Task 11) ----------

def compose_svg_horizontal(
    svgs: list[str],
    *,
    spacing: float = 10.0,
    align: Literal["top", "center", "bottom"] = "top",
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    left_inset: float | None = None,
    right_inset: float | None = None,
    anchor: str | None = None,
) -> str:
    """
    Lay out SVG panels side-by-side.

    When *title*, *subtitle*, and *caption* are all ``None`` (default) the
    output is byte-identical to the previous behavior.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.compose_svg_horizontal([svg1, svg2], spacing=10)
    """
    ...

def compose_svg_vertical(
    svgs: list[str],
    *,
    spacing: float = 10.0,
    align: Literal["left", "center", "right"] = "left",
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    left_inset: float | None = None,
    right_inset: float | None = None,
    anchor: str | None = None,
) -> str:
    """
    Stack SVG panels top-to-bottom.

    When *title*, *subtitle*, and *caption* are all ``None`` (default) the
    output is byte-identical to the previous behavior.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.compose_svg_vertical([svg1, svg2], spacing=10)
    """
    ...

def compose_svg_grid(
    cells: list[str | None],
    *,
    rows: int,
    cols: int,
    row_ratios: list[float],
    col_ratios: list[float],
    spacing: float = 10.0,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    left_inset: float | None = None,
    right_inset: float | None = None,
    anchor: str | None = None,
) -> str:
    """
    Arrange SVG panels in a grid.

    When *title*, *subtitle*, and *caption* are all ``None`` (default) the
    output is byte-identical to the previous behavior.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.compose_svg_grid([[svg1, svg2], [svg3, svg4]], spacing=10)
    """
    ...

# ---------- Continuous color schemes (Phase 8b Task 37) ----------

class ContinuousScheme:
    @staticmethod
    def from_name(name: str) -> "ContinuousScheme": ...
    def reversed(self) -> "ContinuousScheme": ...
    def __repr__(self) -> str: ...

def Gradient(stops: list[tuple[float, str]]) -> ContinuousScheme: ...

# ---------- Diagnostics (Phase 10g Task 35) ----------

class _KendallResult(TypedDict):
    tau: float
    n_concordant: int
    n_discordant: int
    n_tied_x: int
    n_tied_y: int
    n_tied_both: int

def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> _KendallResult: ...

# ---------- Model-diagnostic kernels (Phase 10g) ----------
# Arrow arrays are passed as Any (pyarrow is a runtime dep, not a type dep).
# RecordBatch returns are also typed as Any for the same reason.

def hat_matrix_stats(x_table: Any, y_true: Any, y_pred: Any) -> Any: ...
def studentized_residual_no_x(y_true: Any, y_pred: Any) -> Any: ...
def roc_curve_kernel(y_true: Any, y_score: Any, drop_intermediate: bool) -> Any: ...
def roc_auc(y_true: Any, y_score: Any) -> float: ...
def pr_curve_kernel(y_true: Any, y_score: Any) -> Any: ...
def average_precision(y_true: Any, y_score: Any) -> float: ...
def calibration_kernel(
    y_true: Any,
    y_prob: Any,
    n_bins: int,
    strategy: str,
) -> Any: ...
def confusion_kernel(
    y_true: Any,
    y_pred: Any,
    labels: Any,
    normalize: str,
) -> Any: ...
def prf_at_thresholds(
    y_true: Any,
    y_score: Any,
    thresholds: Any,
) -> Any: ...
