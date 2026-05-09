from typing import Any, Literal, Optional, Union

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
    data: str

    def __init__(
        self,
        *,
        mark: MarkStr,
        x: Union[str, EncodingSpec, None] = None,
        y: Union[str, EncodingSpec, None] = None,
        data: Optional[str] = None,
    ) -> None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "ChartSpec": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
