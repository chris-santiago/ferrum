"""Shared sentinel for distinguishing an explicitly-passed ``title=None`` from an
omitted ``title`` parameter in ``Axis`` and ``Legend`` value objects.

Contract (mirrors ``base.py`` / prepare.rs):
  - title omitted (``_UNSET``) → do not emit the "title" key; Rust falls back to the
    field name.
  - title=None (explicit ``None``) → emit ``title: ""``; Rust treats an empty string as
    "suppress" (no title, no margin).
  - title="Foo" → emit ``title: "Foo"`` verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    pass


class _UnsetType:
    """Singleton sentinel class for the "title not passed" state."""

    _instance: "_UnsetType | None" = None

    def __new__(cls) -> "_UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_UNSET"


#: Sentinel value used as the default for ``Axis.title`` and ``Legend.title``.
#: When a field holds this value, the serializer omits the "title" key entirely,
#: allowing Rust to fall back to the field name.  An explicit ``None`` means
#: "suppress the title".
_UNSET: _UnsetType = _UnsetType()

#: Type alias for the three valid states of the ``title`` parameter.
TitleParam = Union[str, None, _UnsetType]


def serialize_title(title: TitleParam) -> str | None:
    """Convert a ``title`` value to the string to include in a serialized dict.

    Returns ``None`` when the key should be omitted entirely (sentinel ``_UNSET``).
    Returns ``""`` when the title should be suppressed (explicit ``None``).
    Returns the string verbatim otherwise.

    This is the single implementation of the three-way title contract shared
    between ``Axis.to_dict`` and ``Legend.to_dict``.
    """
    if title is _UNSET:
        return None  # omit key — Rust will use the field name
    if title is None:
        return ""  # explicit suppress — Rust treats "" as "no title"
    return title  # type: ignore[return-value]
