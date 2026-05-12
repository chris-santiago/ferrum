"""Theme class, built-in named themes, and default-theme controls."""

from __future__ import annotations

from typing import Any


class Theme:
    """Immutable theme value class for chart styling.

    Pass to ``Chart.theme(t)`` per-chart or activate process-wide via
    ``set_default_theme(t)``. Per-chart ``.theme()`` always wins at render
    time.

    All keys listed in ``ferrum-spec.md`` §3.13 are plumbed end-to-end to
    the Rust renderer. Unknown keys raise ``ValueError`` at construction.

    Parameters
    ----------
    background : str, optional
        Chart background color as a CSS hex string (e.g. ``"#ffffff"``).
        ``background_color`` accepted as an alias.
    mark_color : str, optional
        Default mark fill/stroke color for marks that have no explicit color
        encoding.
    font_family, font_weight, font_color, font_size : optional
        Default text styling for body text (axis titles, etc.).
    title_font_family, title_font_size, title_font_weight, title_color, \
title_anchor, title_offset : optional
        Chart title styling. ``title_anchor`` ∈ {"start", "middle", "end"}.
        Unset values fall back to the corresponding body-text key
        (``title_color`` → ``font_color``, etc.).
    label_font_family, label_color : optional
        Tick label styling. Fall back to ``font_family`` / ``font_color``.
    grid : bool, optional
        Whether to draw grid lines (default True).
    grid_color, grid_width, grid_dash, grid_opacity : optional
        Grid line styling. ``grid_dash`` is a list of dash lengths.
    axis_line : bool, optional
        Whether to draw axis strokes (default True).
    axis_line_color, axis_line_width, tick_color, tick_size, tick_width : optional
        Axis and tick styling.
    point_size, point_opacity, line_stroke_width, bar_corner_radius, \
area_opacity, opacity : optional
        Mark styling.
    color_scheme : str, optional
        Named categorical palette: one of ``okabe_ito`` (default),
        ``tableau10``, ``set1``, ``set2``, ``paired``, ``pastel``, ``dark2``.
        Sequential names (``viridis``, ``plasma``, ``magma``, ``inferno``,
        ``cividis``) also accepted.
    legend_orient, legend_direction, legend_title_font_size : optional
        Legend layout.
    padding, axis_title_padding, column_padding, row_padding : optional
        Spacing in pixels.
    strip_background_color : optional
        Facet strip-title background color.

    Raises
    ------
    ValueError
        If any keyword argument is not in the supported key list (see
        ``Theme._KNOWN_KEYS``).

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.Theme(background="#f9f9f9", grid=False, padding=16)
    >>> t2 = t.update(mark_color="#e74c3c")
    >>> t2
    Theme(background='#f9f9f9', grid=False, mark_color='#e74c3c', padding=16)
    """

    __slots__ = ("_props",)

    _KNOWN_KEYS: frozenset[str] = frozenset(
        {
            # Canvas
            "background",
            "background_color",
            "padding",
            # Typography
            "font_family",
            "font_weight",
            "font_color",
            "font_size",
            "title_font_family",
            "title_font_size",
            "title_font_weight",
            "title_color",
            "title_anchor",
            "title_offset",
            "label_font_family",
            "label_color",
            # Grid
            "grid",
            "grid_color",
            "grid_width",
            "grid_dash",
            "grid_opacity",
            # Axes
            "axis_line",
            "axis_line_color",
            "axis_line_width",
            "tick_color",
            "tick_size",
            "tick_width",
            # Marks
            "mark_color",
            "point_size",
            "point_opacity",
            "line_stroke_width",
            "bar_corner_radius",
            "area_opacity",
            "opacity",
            # Palette
            "color_scheme",
            # Strip
            "strip_background_color",
            # Legend
            "legend_orient",
            "legend_direction",
            "legend_title_font_size",
            # Reference lines
            "reference_line_color",
            "reference_line_dash",
            # Spacing
            "axis_title_padding",
            "column_padding",
            "row_padding",
        }
    )

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - self._KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"Unknown Theme key(s): {sorted(unknown)!r}. "
                f"See ferrum-spec.md §3.13 for the supported key list."
            )
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

    _FALLBACKS: dict[str, str] = {
        "title_color": "font_color",
        "label_color": "font_color",
        "title_font_family": "font_family",
        "label_font_family": "font_family",
    }

    def to_theme_inputs_dict(self) -> dict:
        """Return a dict suitable for ``ferrum._core.render_svg(theme=...)``.

        Resolves spec-defined fallbacks (e.g. ``title_color`` falls back to
        ``font_color`` if unset) and normalises the public Python alias
        ``background`` to the Rust binding's canonical ``background_color``
        key. Rust sees a fully-resolved dict; no Option fallback chains in
        the binding.
        """
        d = dict(self._props)
        # Apply fallback chains BEFORE the background rename so a future
        # fallback whose source is "background" (none today) still resolves.
        for derived, source in self._FALLBACKS.items():
            if derived not in d and source in d:
                d[derived] = d[source]
        # Rust binding reads "background_color"; normalise the public Python
        # alias "background" so every built-in theme renders its configured
        # background regardless of which key was used at construction.
        if "background" in d:
            d["background_color"] = d.pop("background")
        return d

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
    "Theme",
    "default",
    "minimal",
    "dark",
    "publication",
    "economist",
    "fivethirtyeight",
    "solarized_light",
    "solarized_dark",
]

from ferrum.themes._defaults import set_default_theme, get_default_theme, theme_context  # noqa: E402

__all__ += ["set_default_theme", "get_default_theme", "theme_context"]
