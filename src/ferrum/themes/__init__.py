"""Theme class, built-in named themes, and default-theme controls."""
from __future__ import annotations

from typing import Any


class Theme:
    """Immutable theme value class for chart styling.

    Pass to ``Chart.theme(t)`` per-chart or activate process-wide via
    ``set_default_theme(t)``. Per-chart ``.theme()`` always wins at render
    time.

    ``Theme`` accepts arbitrary keyword arguments. Only the keys listed below
    are currently wired to the Rust renderer — all others are stored and
    round-trip through ``update()`` / ``to_theme_inputs_dict()`` but are
    silently ignored at render time (reserved for future phases).

    Parameters
    ----------
    background_color : str, optional
        Chart background color as a CSS hex string (e.g. ``"#ffffff"``).
    mark_color : str, optional
        Default mark fill/stroke color for marks that have no explicit color
        encoding.
    point_size : float, optional
        Default point radius in pixels.
    line_stroke_width : float, optional
        Default line stroke width in pixels.
    bar_corner_radius : float, optional
        Corner radius applied to bar marks.
    area_opacity : float, optional
        Default opacity for area marks.
    grid : bool, optional
        Whether to draw grid lines.
    padding : float, optional
        Chart padding in pixels applied to all four sides.
    **kwargs : any
        Additional styling keys (e.g. ``font_color``, ``title_color``,
        ``color_scheme``, ``font_family``, ``axis_line``, ``grid_color``).
        Stored in the theme and returned by ``to_theme_inputs_dict()`` but
        not yet consumed by the Rust renderer — reserved for future phases.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.Theme(background_color="#f9f9f9", grid=False, padding=16)
    >>> t2 = t.update(mark_color="#e74c3c")
    >>> t2
    Theme(background_color='#f9f9f9', grid=False, mark_color='#e74c3c', padding=16)
    """

    __slots__ = ("_props",)

    def __init__(self, **kwargs: Any) -> None:
        self._props: dict = {k: v for k, v in kwargs.items() if v is not None}

    def update(self, **kwargs: Any) -> "Theme":
        """Return a new ``Theme`` with the given properties overridden.

        Passing ``None`` for a key removes that property from the derived
        theme. The source theme is unchanged (immutable).

        Parameters
        ----------
        **kwargs : any
            Property overrides. Pass ``None`` to clear a property.

        Returns
        -------
        Theme
            A new ``Theme`` with the merged properties.

        Examples
        --------
        >>> import ferrum as fm
        >>> base = fm.Theme(grid=False, padding=12)
        >>> with_bg = base.update(background_color="#222")
        """
        merged = {**self._props}
        for k, v in kwargs.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        return Theme(**merged)

    def to_theme_inputs_dict(self) -> dict:
        """Return a dict suitable for ``ferrum._core.render_svg(theme=...)``."""
        return dict(self._props)

    def __eq__(self, other: object) -> bool:
        """Return True if *other* is a ``Theme`` with identical properties."""
        if not isinstance(other, Theme):
            return NotImplemented
        return self._props == other._props

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys."""
        # Use repr(v) so non-hashable values (e.g. grid_dash=[5, 3]) don't break hashing.
        # Mirrors ChannelBase.__hash__ pattern.
        return hash(tuple(sorted((k, repr(v)) for k, v in self._props.items())))

    def __repr__(self) -> str:
        """Return a constructor-style string representation."""
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

from ferrum.themes._defaults import set_default_theme, get_default_theme, theme_context  # noqa: E402
__all__ += ["set_default_theme", "get_default_theme", "theme_context"]
