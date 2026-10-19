import datetime
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
    Round-trip an Arrow C Data Interface stream through the Rust transport.

    Reads any object exposing ``__arrow_c_stream__`` (a polars
    ``DataFrame``, a pyarrow ``Table`` or ``RecordBatch``) and returns a
    one-shot ``RecordBatchReader`` over the same batches with the first
    column renamed to ``<name>_renamed``. It checks the zero-copy
    transport path and is not part of chart rendering. Raises
    ``ValueError`` when the input has zero columns.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    >>> pl.from_arrow(fm.process_batch(df)).columns
    ['x_renamed', 'y']
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
    #: The class a binned color was written with ("Color", "Fill" or "Stroke").
    #: Accepted only beside a binned color scale (a ``bin-ordinal`` scale with
    #: ``bin``); without one it is refused, at construction and at render.
    bin_channel: Optional[str]
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
        bin_channel: Optional[Literal["Color", "Fill", "Stroke"]] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

# `layers` below is a list of per-layer wire dicts (see `crate::spec::layer`):
#   {"mark": str, "encoding": dict,
#    # every key optional, each omitted when unset so a spec that uses none
#    # serializes exactly as a spec predating it:
#    "mark_style"?: dict, "transforms"?: list, "data_source"?: str,
#    "position"?: dict, "name"?: str, "independent_y"?: bool, "coord"?: dict}
# A layer refuses any other key. Its blend mode is `mark_style["blend"]`.
# `coord` (per-layer coordinate system, GH #172(1)) takes the same
# internally-tagged `{"kind": ...}` dict the chart-level `coord` accepts, and
# places THIS layer's marks: a projection (`{"kind": "polar"|"geo"}`) or the
# channel exchange (`{"kind": "flip"}`). Absent means "inherit the chart's".
# Unlike the chart-level slot it does NOT accept a bare `"flip"` token — the
# layer dict is deserialized by serde, with no constructor to widen one
# (`ferrum.coord.coord_wire` normalizes on the Python side). A layer coord that
# declares a PANEL property instead of a placement (`x_domain`/`y_domain`,
# `expand: false`, `clip: false`, or `{"kind": "fixed"}`) is refused at render:
# the layers share one panel, so one of them cannot have its own domain or
# aspect ratio.
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
        entry: Optional[str] = None,
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
    reverse: bool
    def __init__(
        self,
        *,
        domain: Sequence[float] | None = None,
        range: Sequence[float] | None = None,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
        reverse: bool = False,
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
    reverse: bool
    def __init__(
        self,
        *,
        domain: Sequence[float] | None = None,
        range: Sequence[float] | None = None,
        base: float = 10.0,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
        reverse: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class TimeScale:
    domain: list[float] | None
    range: list[float] | None
    clamp: bool
    padding: float | None
    utc: bool
    reverse: bool
    def __init__(
        self,
        *,
        domain: Sequence[float | datetime.date | datetime.datetime | str] | None = None,
        range: Sequence[float] | None = None,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
        utc: bool = False,
        reverse: bool = False,
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
    reverse: bool
    def __init__(
        self,
        *,
        domain: Sequence[float] | None = None,
        range: Sequence[float] | None = None,
        constant: float = 1.0,
        clamp: bool = False,
        nice: bool = False,
        padding: float | None = None,
        reverse: bool = False,
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
    reverse: bool
    def __init__(
        self,
        *,
        domain: Optional[Sequence[float]] = None,
        range: Optional[Sequence[float]] = None,
        exponent: float = 2.0,
        clamp: bool = False,
        nice: bool = False,
        padding: Optional[float] = None,
        reverse: bool = False,
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
    reverse: bool
    def __init__(
        self,
        *,
        domain: Optional[Sequence[float]] = None,
        range: Optional[Sequence[float]] = None,
        clamp: bool = False,
        nice: bool = False,
        padding: Optional[float] = None,
        reverse: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "SqrtScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class BandScale:
    """Discrete band scale for bar charts.

    ``padding``, ``padding_inner`` and ``padding_outer`` are each ``None``
    until the caller sets them. With none set, bands lay out exactly as the
    auto-inferred categorical axis does (no inner or outer padding), not
    Vega-Lite's 0.1 convention. An unset side falls back to ``padding``,
    then to 0 (d3's own behavior), so ``padding_inner=0.2`` alone leaves
    outer padding at 0, not 0.1. The getters mirror this: ``padding`` returns
    the user-set value or ``None``; ``padding_inner``/``padding_outer``
    return ``side ?? padding``, or ``None`` when neither is set. An explicit
    ``0.0`` is distinct from ``None`` in equality and ``repr()``, even though
    both resolve to the same rendered geometry.
    """

    domain: list[str]
    range: Optional[list[float]]
    padding: Optional[float]
    padding_inner: Optional[float]
    padding_outer: Optional[float]
    align: float
    def __init__(
        self,
        *,
        domain: Optional[Sequence[str]] = None,
        padding: Optional[float] = None,
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
    """Discrete point scale for dot plots.

    ``padding`` defaults to ``None`` in the constructor, and the ``padding``
    getter returns it as passed: ``None`` when unset, as ``BandScale``,
    ``LinearScale`` and ``TimeScale`` do (``OrdinalScale`` is the exception:
    its constructor defaults ``padding`` to ``0.0`` and its getter returns
    that). The renderer owns the default for an unset padding (half
    a step at each end). Equality and ``repr()`` see the same value, so
    ``PointScale() != PointScale(padding=0.5)`` and ``repr(PointScale())``
    prints ``padding=None``, even though both render identically.
    """

    domain: list[str]
    range: Optional[list[float]]
    padding: Optional[float]
    align: float
    reverse: bool
    def __init__(
        self,
        *,
        domain: Optional[Sequence[str]] = None,
        padding: Optional[float] = None,
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
        bins: Optional[Union[str, int, float]] = None,
        bin_count: Optional[int] = None,
        bin_width: Optional[float] = None,
        extent: Optional[Tuple[float, float]] = None,
        nice: bool = True,
        cumulative: bool = False,
        shared_extent: bool = False,
        groupby: Optional[Union[str, List[str]]] = None,
        agg_field: Optional[str] = None,
        agg: Optional[str] = None,
        per_row: bool = False,
        aggregates: Optional[List[AggregateOp]] = None,
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
        groupby: Optional[Union[str, List[str]]] = None,
        count_scale: Optional[Union[str, int, float]] = None,
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
        groupby: Optional[Union[str, List[str]]] = None,
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
        bandwidth: Optional[float] = None,
        degree: Optional[int] = None,
        k: Optional[int] = None,
        n: int = 200,
        seed: int = 0,
        x_bins: Optional[int] = None,
        x_estimator: Optional[str] = None,
        output: str = "fitted",
        inject_zero_ref: bool = False,
        inject_metrics: bool = False,
        groupby: Optional[Union[str, List[str]]] = None,
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
        groupby: List[str] = ...,
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
        point_size: float = 4.0,
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
        axis: str = "rows",
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
        groupby: List[str] = ...,
        k_depth: str = "proportion",
        k_proportion: float = 0.007,
        outlier_threshold: float = 0.0,
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
        groupby: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
    ) -> None: ...

# ---------- Data transforms (Vega-lite-style, declared missing from stub) ----------
#
# The eight Phase-12 ``Data*`` transforms (DataAggregate / DataBin /
# DataCalculate / DataFilter / DataFold / DataPivot / DataStack / DataWindow)
# have no Python class. They are constructed exclusively via the dict-emitting
# ``ferrum.transform_*`` functions (see ``ferrum/transforms.py``) and carried to
# Rust through the ``transforms_json`` serde path (SEAM-02). The typed pyclasses
# that used to be declared here were never imported or exported, so they were
# removed.

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
        groupby: Optional[Union[str, List[str]]] = None,
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

    The tick-label lists are the labels the axes will draw, which size the
    margins. Returns a dict with ``"viewport"``, ``"panels"`` and ``"axes"``.

    Examples
    --------
    >>> import ferrum as fm
    >>> spec = fm.ChartSpec(mark="point", x="a", y="b")
    >>> layout = fm.compute_layout(
    ...     spec,
    ...     viewport=(300.0, 200.0),
    ...     x_tick_labels=["1", "2", "3"],
    ...     y_tick_labels=["1", "2", "4"],
    ... )
    >>> sorted(layout)
    ['axes', 'panels', 'viewport']
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

    ``data`` is any Arrow C Data Interface source (a polars ``DataFrame``,
    a pyarrow ``Table``). Layout is computed internally from ``viewport``.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 1.0, 2.0]})
    >>> spec = fm.ChartSpec(mark="point", x="a", y="b")
    >>> svg = fm.render_svg(spec, df, viewport=(300.0, 200.0))
    >>> svg.startswith("<svg")
    True
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

    Takes the same arguments as ``render_svg``.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 1.0, 2.0]})
    >>> spec = fm.ChartSpec(mark="point", x="a", y="b")
    >>> png = fm.render_png(spec, df, viewport=(300.0, 200.0))
    >>> png[1:4]
    b'PNG'
    """
    ...

# ---------- Theme key manifest (single source of truth) ----------
# The complete valid theme-key set and its color-typed subset, published from
# the Rust `ThemeOverridesSpec`. The Python `Theme` derives `_KNOWN_KEYS` from
# `theme_known_keys()` so the two contracts cannot drift.

def theme_known_keys() -> list[str]: ...
def theme_color_keys() -> list[str]: ...

# #123: the aux-legend channel names (``size``,
# ``shape``, ``stroke_dash`` — the channels whose ``legend=`` shares the
# chart's single legend surface instead of getting its own), published from
# Rust's ``AUX_LEGEND_CHANNEL_NAMES`` so ``ferrum.encoding.base`` reads the
# list instead of hand-mirroring it as a Python literal.
def aux_legend_channels() -> list[str]: ...

# The channels Rust builds a panel's auxiliary (non-positional) scale for, in
# its ``AuxiliaryScales`` order, each filled from the first layer that draws
# the channel and resolved over every layer that draws it.
def auxiliary_scale_channels() -> list[str]: ...

# Every ``MarkKwargsSpec`` wire key (the ``mark_style`` dict's keys), in
# declaration order, read from the struct's derived serde field list.
def mark_kwargs_fields() -> list[str]: ...

# The numeric (``f64``/``usize``) subset of ``mark_kwargs_fields()``, in
# declaration order, derived from the struct's own ``Deserialize``.
def mark_numeric_kwargs() -> list[str]: ...

# The ``usize`` subset of ``mark_numeric_kwargs()``, in declaration order.
# Partitions the numeric fields exactly with ``mark_numeric_kwargs()`` minus
# this list: route these through a non-negative-integer contract, not the
# real-number one every other numeric field gets.
def mark_integer_kwargs() -> list[str]: ...

# The ``bool`` subset of ``mark_kwargs_fields()``, in declaration order,
# derived from the struct's own ``Deserialize``: a field is boolean exactly
# when it accepts a bare ``true`` under its key.
def mark_bool_kwargs() -> list[str]: ...

# The string-shaped subset of ``mark_kwargs_fields()``, in declaration
# order, derived from the struct's own ``Deserialize``: a field matches
# exactly when it accepts the bare string ``"x"`` under its key.
def mark_string_kwargs() -> list[str]: ...

# The marks that draw a constant ``size=``, from Rust's one per-mark size
# table (``Mark::size_drawing``). Every other mark refuses a constant size.
def marks_drawing_constant_size() -> list[str]: ...

# The subset of ``offered`` (inherited positional channels, wire names ``x``,
# ``y``, ``x2``, ``y2``) a layer of primitive ``mark`` takes, given the
# positional channels ``own`` it writes itself, in wire order: from Rust's one
# per-mark table (``Mark::inherited_positional``). A span channel is taken
# only by a mark that draws it; a rule or tick writing exactly one of x/y and
# no span takes no other. An unknown mark or channel raises ``ValueError``.
def mark_inherited_positional(mark: str, own: list[str], offered: list[str]) -> list[str]: ...

# Whether a layer of primitive ``mark`` takes an inherited binding of the
# non-positional wire channel ``channel`` as a binding it draws: from Rust's
# one per-mark table (``Mark::draws_inherited``). False only for a text's
# ``color``, which a text draws only when it binds it; an inherited color
# reaches a text only as its grouping key. An unknown mark or channel raises
# ``ValueError``.
def mark_draws_inherited(mark: str, channel: str) -> bool: ...

# The non-positional wire channels primitive ``mark`` draws when bound to a
# field, in ``Encoding`` declaration order: from Rust's one per-mark table
# (``Mark::draws_channel``). A binding on any other channel is never read by
# the mark's builder. An unknown mark raises ``ValueError``.
def mark_drawn_channels(mark: str) -> list[str]: ...

# The columns ``bin`` computes, from Rust's one Bin column builder:
# ``bin_start``, ``bin_end``, ``count``, ``density``, then ``agg_value`` when
# ``agg_field``/``agg`` is set, else each ``aggregates`` entry's ``as`` in
# entry order; under ``per_row`` only ``bin_start``, ``bin_end``. Groupby
# columns are never listed. A non-``Bin`` argument raises ``TypeError``.
def bin_output_columns(bin: Bin) -> list[str]: ...

# The columns a layer of primitive ``mark`` draws for its binding ``field`` on
# ``channel``: a raster image's extent columns on ``x``/``y`` and its value
# column on ``color``, else ``[field]``. ``has_url`` says whether an image
# layer binds ``url`` (the url-tile path, which draws its own fields). From
# Rust's one drawn-column answer; an unknown mark raises ``ValueError``.
def drawn_positional_columns(mark: str, has_url: bool, channel: str, field: str) -> list[str]: ...

# Run the render-time constant-size rule early: raise ``ValueError`` with the
# render refusal's exact text, voiced as ``entry``, when ``mark`` never draws a
# constant size or ``size`` is not a finite number above zero. ``mark`` must be
# a primitive mark name; any other name raises ``ValueError``.
def check_mark_size(entry: str, mark: str, size: float) -> None: ...

# The constant-size rule's reason clause for ``size`` on ``mark``, naming no
# call and with no trailing period, from the same source as
# ``check_mark_size``'s refusal text; ``None`` when the rule accepts ``size``.
# ``mark`` must be a primitive mark name; any other name raises ``ValueError``.
def mark_size_reason(mark: str, size: float) -> str | None: ...

# The numeric columns of ``schema`` (any object exporting
# ``__arrow_c_schema__``), in schema order: exactly the integer and float
# columns ``Linkage`` clusters and ``Reorder(axis="columns")`` permutes. A
# decimal column raises ``ValueError`` voiced as ``entry``, naming the column
# and its dtype.
def numeric_columns(schema: Any, entry: str) -> list[str]: ...

# Whether an undeclared encoding over ``field`` of ``schema`` (any object
# exporting ``__arrow_c_schema__``) reads categorical to the render. The schema
# must be ``ferrum._coerce.normalize_for_rust`` output, the dtypes the render
# receives; the reading applies no cast of its own. Over those dtypes a string
# or boolean column reads categorical, a number or timestamp does not, and a
# field the schema lacks reads ``False``.
def column_reads_categorical(schema: Any, field: str) -> bool: ...

# The refusal text of the render gate's unknown-column error
# (``RenderError::LayerUnknownColumn``), formatted by its own ``Display``, so a
# Python check that knows a layer's schema before the render raises the gate's
# words. ``channel`` is any channel name ``encode()`` accepts, aliases
# included (``x``, ``color``, ``fill``, ``detail``, ``order``, a facet
# channel, ...; ``ferrum.encoding._channel_policy._SPEC_KNOWN_CHANNELS``), and
# the message names it as written; ``remedy`` is ``"plain"``, ``"source"`` or
# ``"statistic"``; ``independent_y`` picks the remedy's spelling for a layer
# with its own y axis; ``entry`` names a layer a figure or composite mark built
# by that public call in place of ``layer {layer}``. Any other ``channel``, or
# an unknown ``remedy``, raises ``ValueError``.
def layer_unknown_column_message(
    layer: int,
    channel: str,
    name: str,
    remedy: str,
    independent_y: bool,
    *,
    entry: str | None = None,
) -> str: ...

# #176: the ``interpolate=`` vocabulary, published from Rust's
# ``Interpolate::ALL`` (in its order) so every ``interpolate=`` door gates it
# at declaration instead of mirroring the list as a Python literal.
def interpolate_accepted_values() -> list[str]: ...

# Rust's per-mark style table (``Mark::style_fields``): each ``mark_style``
# field ``mark`` draws, mapped to the input it needs to draw (a
# ``StyleNeed`` name such as ``"stroke"``), or ``None`` when it draws
# whenever set. Raises ``ValueError`` for a name that is no primitive mark.
def mark_style_fields(mark: str) -> dict[str, Optional[str]]: ...

# The marks ``mark_style_fields`` has a row for, in table order, so Python
# reads its mark population from here rather than keeping a list of its own.
def style_table_marks() -> list[str]: ...

# The ``blend=`` vocabulary (``BlendMode``, with its ``"alpha"`` alias of
# ``"normal"``), in Rust's order.
def blend_accepted_values() -> list[str]: ...

# The ``stroke_cap=`` vocabulary, in Rust's order.
def stroke_cap_accepted_values() -> list[str]: ...

# The ``stroke_join=`` vocabulary, in Rust's order.
def stroke_join_accepted_values() -> list[str]: ...

# #137: the ``Smooth(x_estimator=)`` vocabulary, published from
# Rust's ``XEstimator`` (a declared subset of the aggregate vocabulary, in its
# refusal-message order) so Python reads it instead of mirroring it.
def x_estimator_accepted_values() -> list[str]: ...

# #176(11) + N3: the ``sort=`` object-form vocabulary (``{"field": ...,
# "op": ..., "order": ...}``), published from the same ordinal-domain sort
# resolution Rust uses to order a category axis, so every Python channel
# whose honored set contains ``sort`` (X, Y, Color, Shape, StrokeDash,
# Facet, FacetRow, FacetCol) and ``EncodingSpec`` gate an object-form sort
# against it at construction instead of mirroring the list as a Python
# literal. Keys are ``"op"`` (e.g. ``"sum"``, ``"mean"``, ``"count"``) and
# ``"order"`` (``"ascending"``, ``"descending"``).
def sort_accepted_vocabulary() -> dict[str, list[str]]: ...

# #165(1): the one home of the ``"{name}_{key}"`` secondary-
# output naming rule (``qq``, ``letter_value``, ``linkage``).
# ``secondary_key(None, k)`` returns ``k`` unprefixed;
# ``secondary_key(n, k)`` returns ``f"{n}_{k}"``.
def secondary_key(name: str | None, key: str) -> str: ...

# Batch C task 4 round 4 (#95): the raw-dict scale key gate's own per-type
# accepted-key table, published for Python to filter against instead of
# hand-mirroring a subset of it. Raises ValueError naming the type on an
# unknown ``scale_type`` tag.
def scale_accepted_keys(scale_type: str) -> list[str]: ...

# The canonical stroke-dash pattern palette
# (``ferrum_scene::DASH_PALETTE``), index-aligned with the Rust
# ``resolve_stroke_dash``/``stroke_dash_index`` pair: index 0 is solid
# (an empty list); index ``i`` (``i >= 1``) is ``DASH_PALETTE[i - 1]``.
# Published so ``_LINETYPE_MAP`` (``marks/base.py``) can assert every one
# of its values, including ``"solid"``'s empty pattern, is a palette member
# without holding a Rust-side copy of the table.
def stroke_dash_palette() -> list[list[float]]: ...

# #159: ``ScaleSpec::TAGS`` -- the full set of ``ScaleSpec`` "type" tags (the
# image of ``ScaleSpec::tag()`` over every variant, kept in step with it by a
# Rust test), in ``TAGS`` order. Published so Python's override registry
# holds no hand-listed mirror of it.
def scale_type_tags() -> list[str]: ...

# The temporal subset of ``scale_type_tags()`` -- the wire tags whose
# ``domain`` is temporal (``"time"``, ``"utc"``).
def scale_temporal_type_tags() -> list[str]: ...

# Batch C close remediation: the same wire gate ``EncodingSpec(field,
# scale={...})`` enforces, callable for validation only. Raises the gate's own
# ValueError (byte-identical to the EncodingSpec route's message, since both
# call one shared Rust helper) on an invalid scale dict; returns None on a
# valid one. Exists so Python's override-scale merge can ask "is this scale
# dict valid?" without constructing a throwaway EncodingSpec for the side
# effect of its exception.
def validate_scale_dict(scale: dict) -> None: ...

# The numeric auxiliary scale gate (size, opacity, fill_opacity,
# stroke_opacity, stroke_width) run on a scale dict with no data, so a channel
# class refuses at construction with the text a wire spec refuses with at
# render: a non-linear type, ``padding``, ``scheme``, or a domain or range
# whose length is not 2. Returns None when the channel admits the scale.
def validate_numeric_aux_scale(channel: str, scale: dict) -> None: ...

# #157: the data-free value bounds the scale classes enforce at
# construction, through the same Rust predicates and so the same clause text:
# band padding (d3's bounds) and ``align`` in [0, 1]; point ``padding`` finite
# and >= 0 and ``align`` in [0, 1]; ordinal ``padding`` in [0, 1]. Callable for
# validation only. Raises ``ValueError`` whose message is the bare clause --
# no ``"scale: "`` prefix, no channel, no call name, no key -- so a caller
# names the call and the path itself, one key at a time (``Chart.override``'s
# per-leaf refusal does exactly this). Returns None on a valid scale dict.
def validate_scale_values(scale: dict) -> None: ...

# The padding-family keys (``padding``, ``paddingInner``, ``paddingOuter``,
# ``align``) a scale type accepts: the keys ``translate_scale_switch``
# translates by meaning. Published so a caller selecting a scale's padding
# family reads this set instead of restating it. Raises ValueError for an
# unknown tag, like ``scale_accepted_keys``.
def scale_padding_keys(scale_type: str) -> list[str]: ...

# A channel's scale dict carried across a switch from ``old_type`` to
# ``new_type``, and the keys the switch drops: the one owner of what a
# scale-type switch keeps. The padding-family keys the old type accepts
# (``padding``, ``paddingInner``, ``paddingOuter``, ``align``) are translated
# by meaning (an ordinal inner gap becomes ``paddingInner``, a point's end
# padding becomes ``paddingOuter``, a band's inner or outer side becomes
# ordinal or point ``padding``) and dropped when the new type gives them no
# meaning; a ``None`` value is unset and writes no key. Every other key the
# old type accepts survives only when the new type accepts it too. A key the
# old type does not accept (``"type"`` included) passes through for the scale
# gate to judge, unless the switch writes a key of that name, which raises
# ValueError naming the key. ``dropped`` lists, in ``base`` order, every key
# of ``base`` whose value the switch does not write onto the new scale, so a
# caller can validate exactly those under the old type. Carried values are
# never checked: a malformed one is left for ``validate_scale_values``. The
# same type returns a copy of ``base`` and drops nothing. Raises ValueError
# for an unknown tag on either side, like ``scale_accepted_keys``.
def translate_scale_switch(old_type: str, new_type: str, base: dict) -> tuple[dict, list[str]]: ...

# Wave-4 task 9 (#193): the ``bins=`` vocabulary gate — the same parser
# ``Bin(bins=)``, ``Bin2D(bins_x=)`` and ``Kde(count_scale=)`` construct
# through, callable for validation only. Raises ValueError prefixed with
# ``ctx`` (the public entry name the caller invoked, e.g. ``"displot"``) on a
# malformed value; returns None on a valid one. Exists so a figure entry can
# refuse in its own voice without a second copy of the accepted spellings
# living in Python.
def validate_bin_rule(bins: Any, ctx: str) -> None: ...

# ---------- Palette registry accessors ----------
#
# Read-only views over the Rust palette registry (categorical + continuous).
# ``ferrum.color`` consumes these as the single source of truth instead of
# hand-mirroring hex tables.

def list_palettes() -> list[str]: ...
def palette_kind(name: str) -> Optional[str]: ...
def palette_colors(name: str) -> Optional[list[str]]: ...
def palette_sample(name: str, t: float) -> Optional[str]: ...

# Batch A Task 1 (F-L02-01/NF-A1): the one full-CSS color parser exposed to
# Python. Parses a CSS name / hex / rgb()/rgba() string and returns
# normalized hex; raises ValueError naming the accepted forms on failure.
def parse_color_to_hex(s: str) -> str: ...

# The same parser's raw-byte-tuple form, for a caller that needs
# (r, g, b, a) directly rather than round-tripping through
# parse_color_to_hex and re-slicing the hex string. Raises ValueError
# naming the accepted forms on failure, identically to parse_color_to_hex.
def parse_color_to_rgba(s: str) -> tuple[int, int, int, int]: ...

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

# `tree` is a nested-dict composite tree (see crate::spec::composite):
#   leaf:      {"kind": "leaf", "spec": ChartSpec, "data": int,
#               # optional Task 5d per-leaf binding overrides (absent = inherit
#               # the call-level value); optional per-child label:
#               "theme"?: dict, "viewport"?: tuple[float, float],
#               "config"?: dict, "chart_config"?: dict, "label"?: str}
#   composite: {"kind": "composite", "layout": str, "children": [...],
#               # root-only chrome: "title"/"subtitle"/"caption"/"config";
#               # per-child (non-root) label:
#               "label"?: str, "resolve"?: dict, "ncols"?: int, "nrows"?: int,
#               "row_ratios"?: list, "col_ratios"?: list, "spacing"?: float,
#               "extent_alignments"?: list}
#               # resolve maps channel -> "shared"/"independent"; supported
#               # channels are "x", "y", "color", "size" (10-pre-b).
#               # extent_alignments (grid only, GH #182 follow-up) is a list of
#               # {"follower": int, "leader": int, "axis": "x"|"y"}: the
#               # follower cell's plot extent spans the leader cell's on that
#               # axis, keeping its own on the other. Cell positions are
#               # row-major in `children` (holes counted). A jointplot's
#               # marginals name its center panel; a clustermap's dendrograms
#               # name its heatmap. Omit (or pass []) for every other grid.
#   hole:      {"kind": "hole"}  # Task 8a: a field-less placeholder cell —
#               # valid only as a direct child of a "grid"/"wrap" composite
#               # (never at the tree root); reserves its row/column slot but
#               # renders no panel/label/chrome. Covers JointChart's empty 2x2
#               # corner and RepeatChart's `corner=True`.
# `theme`/`config`/`chart_config`/`viewport` below are the call-level defaults
# every leaf inherits unless it carries its own override.
def render_composite_svg(
    tree: Any,
    payloads: Any,
    *,
    viewport: Any,
    theme: Any = None,
    config: Any = None,
    chart_config: Any = None,
) -> str: ...
def render_composite_interactive(
    tree: Any,
    payloads: Any,
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

# ---------- Figure chrome (flat single-chart caption/title band) ----------

def wrap_svg_with_chrome(
    svg: str,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    left_inset: float | None = None,
    right_inset: float | None = None,
    anchor: str | None = None,
) -> str:
    """
    Wrap a single already-rendered SVG with a figure-level chrome band.

    Private ``ferrum._core`` implementation detail of ``Chart.to_svg()``'s
    ``.properties(caption=...)`` post-wrap. When *title*, *subtitle*, and
    *caption* are all ``None`` the output is byte-identical to the input.
    """
    ...

# ---------- Continuous color schemes (Phase 8b Task 37) ----------

class ContinuousScheme:
    def __init__(self, name: str) -> None: ...
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

# ---------- Transform registry (Phase batch-D task 1, spec §4.6) ----------

class _TransformRegistryEntry(TypedDict):
    methods: list[str]
    methods_needing_field: list[str]
    outputs: list[tuple[str, str]]
    secondary_outputs: list[tuple[str, list[tuple[str, str]]]]
    params: list[str]

def transform_registry(kind: str) -> _TransformRegistryEntry:
    """
    ``ferrum._core.transform_registry(kind)`` — spec §4.6, batch-D task 1.

    Returns ``{"methods": [...], "methods_needing_field": [...],
    "outputs": [(name, dtype), ...],
    "secondary_outputs": [(key, [(name, dtype), ...]), ...], "params": [...]}``
    for a known transform kind (module name — see the module doc); raises
    ``ValueError`` listing every known kind otherwise.

    ``secondary_outputs`` (2026-09-06) declares the additional NAMED batches a
    kind publishes alongside its primary rows — ``qq``'s reference line,
    ``letter_value``'s outliers and per-depth bands, ``linkage``'s dendrogram
    order/coords/segments. Keys are the ones used under default construction
    (no ``name=``); a spec ``name`` prefixes each as ``"{name}_{key}"``.
    """
    ...
