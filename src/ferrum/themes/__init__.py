"""Theme value class + 8 builtins + set_default_theme."""
from __future__ import annotations

from typing import Any


class Theme:
    """Immutable theme value class. Pass via Chart.theme(t) or set_default_theme(t).

    All props default to None and are dropped from the dict passed to the
    Rust ThemeInputs binding (so Rust falls back to its defaults).

    Use .update(**kwargs) to derive a new Theme with overrides; the source
    theme is unchanged.
    """

    __slots__ = ("_props",)

    def __init__(self, **kwargs: Any) -> None:
        self._props: dict = {k: v for k, v in kwargs.items() if v is not None}

    def update(self, **kwargs: Any) -> "Theme":
        merged = {**self._props}
        for k, v in kwargs.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        return Theme(**merged)

    def to_theme_inputs_dict(self) -> dict:
        """Return a dict suitable for ferrum._core.render_svg(theme=...)."""
        return dict(self._props)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Theme):
            return NotImplemented
        return self._props == other._props

    def __hash__(self) -> int:
        # Use repr(v) so non-hashable values (e.g. grid_dash=[5, 3]) don't break hashing.
        # Mirrors ChannelBase.__hash__ pattern.
        return hash(tuple(sorted((k, repr(v)) for k, v in self._props.items())))

    def __repr__(self) -> str:
        if not self._props:
            return "Theme()"
        kv = ", ".join(f"{k}={v!r}" for k, v in sorted(self._props.items()))
        return f"Theme({kv})"


from ferrum.themes import builtins as _builtins  # noqa: E402

# Re-export the 8 builtins as module attributes
default = _builtins.default
minimal = _builtins.minimal
dark = _builtins.dark
publication = _builtins.publication
economist = _builtins.economist
fivethirtyeight = _builtins.fivethirtyeight
solarized_light = _builtins.solarized_light
solarized_dark = _builtins.solarized_dark

__all__ = [
    "Theme", "default", "minimal", "dark", "publication", "economist",
    "fivethirtyeight", "solarized_light", "solarized_dark",
]
