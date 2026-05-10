from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

DataTypeStr = Literal[
    "Q", "N", "O", "T",
    "quantitative", "nominal", "ordinal", "temporal",
]
MarkStr = Literal[
    "point", "line", "bar", "area", "rule", "text", "tick", "rect",
]


def process_batch(data: Any) -> Any: ...


class EncodingSpec:
    field: str
    type_: Optional[str]
    def __init__(self, field: str, type_: Optional[DataTypeStr] = None) -> None: ...
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
        data: Optional[str] = None,
        transforms: Optional[List[object]] = None,
        facet: Optional[dict] = None,
        layers: Optional[List[dict]] = None,
        coord: Optional[Literal["cartesian", "flip"]] = None,
        mark_style: Optional[dict] = None,
    ) -> None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "ChartSpec": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


# ---------- Scales (Phase 4) ----------

class LinearScale:
    domain: list[float]
    range: list[float]
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LinearScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class LogScale:
    domain: list[float]
    range: list[float]
    base: float
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        base: float = 10.0,
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "LogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class TimeScale:
    domain: list[float]
    range: list[float]
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "TimeScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class SymlogScale:
    domain: list[float]
    range: list[float]
    constant: float
    clamp: bool
    def __init__(
        self,
        *,
        domain: Sequence[float],
        range: Sequence[float],
        constant: float = 1.0,
        clamp: bool = False,
        nice: bool = False,
    ) -> None: ...
    def scale(self, x: float) -> float: ...
    def invert(self, y: float) -> float: ...
    def ticks(self, count: int = 10) -> list[float]: ...
    def nice(self) -> "SymlogScale": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


class OrdinalScale:
    domain: list[str]
    range: list[float]
    padding: float
    def __init__(
        self,
        *,
        domain: Sequence[str],
        range: Sequence[float],
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
    ) -> None: ...

class Kde:
    def __init__(
        self,
        field: str,
        *,
        bandwidth: object = "scott",   # str ("scott"|"silverman") or float
        n: int = 512,
        extent: Optional[Tuple[float, float]] = None,
        cumulative: bool = False,
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
    ) -> None: ...

class AggregateOp:
    def __init__(self, field: str, fn_: str, as_: str) -> None: ...

class Aggregate:
    def __init__(self, ops: List[AggregateOp], *, groupby: Optional[List[str]] = None) -> None: ...

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
) -> dict: ...


def render_svg(
    spec: ChartSpec,
    data: Any,
    *,
    viewport: tuple[float, float],
    theme: Optional[dict] = None,
    config: Optional[dict] = None,
) -> str: ...


def render_png(
    spec: ChartSpec,
    data: Any,
    *,
    viewport: tuple[float, float],
    theme: Optional[dict] = None,
    config: Optional[dict] = None,
) -> bytes: ...


# ---------- SVG compositor (Phase 8a Task 11) ----------

def compose_svg_horizontal(
    svgs: list[str],
    *,
    spacing: float = 10.0,
    align: Literal["top", "center", "bottom"] = "top",
) -> str: ...

def compose_svg_vertical(
    svgs: list[str],
    *,
    spacing: float = 10.0,
    align: Literal["left", "center", "right"] = "left",
) -> str: ...
