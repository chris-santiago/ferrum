"""The ``Theme`` value class — leaf module with no intra-package imports.

This module is the root of the ``ferrum.themes`` dependency graph. It imports
only the standard library and ``ferrum._core`` (the Rust extension), never
anything from the ``ferrum.themes`` package itself. ``builtins`` and
``_defaults`` import ``Theme`` from here; ``ferrum.themes.__init__`` re-exports
``Theme`` (and the builtins and default-theme controls) as a top-of-module
façade. Keeping ``Theme`` in a leaf module is what lets the package use plain
top-level imports instead of the statement-ordered mid-module imports that the
old single-file layout required.
"""

from __future__ import annotations

from typing import Any

from ferrum._core import theme_known_keys as _theme_known_keys


class Theme:
    """Immutable theme value class for chart styling.

    Pass to ``Chart.theme(t)`` per-chart or activate process-wide via
    ``set_default_theme(t)``. Per-chart ``.theme()`` always wins at render
    time.

    **Cascade behavior:** ``Chart.theme(t)`` replaces the entire active theme
    with ``t`` — it does not merge individual keys. To override a subset of
    keys on top of an existing theme, use ``Theme.update()``::

        base = fm.themes.paper_ink
        custom = base.update(grid=False, padding=20)
        chart.theme(custom)

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
        Default body-text styling. ``font_size`` sets the label/body font
        size (the Rust binding's ``label_font_size``), the same way
        ``background`` is the public alias for ``background_color``.
    title_font_family, title_font_size, title_font_weight, title_color, \
title_anchor, title_offset : optional
        Chart title styling. ``title_anchor`` ∈ {"start", "middle", "end"}.
        Unset ``title_*`` / ``label_*`` keys fall back to their body-text
        counterpart (see the "Fallbacks" list below).
    label_font_family, label_color : optional
        Tick label styling (see the "Fallbacks" list below).
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
        Named categorical palette: one of ``paper_ink`` (default),
        ``slate_citrus``, ``arctic_signal``, ``okabe_ito``,
        ``tableau10``, ``set1``, ``set2``, ``paired``, ``pastel``, ``dark2``.
        Sequential names (``viridis``, ``plasma``, ``magma``, ``inferno``,
        ``cividis``, ``blues``, ``reds``, ``greens``, ``oranges``,
        ``purples``, ``cool_blue``, ``warm_ochre``, ``night_blue``,
        ``electric_lime``, ``signal_blue``, ``ember_orange``) and diverging
        names (``rdbu``, ``blue_to_red``, ``cyan_to_amber``,
        ``blue_to_violet``) also accepted.
    sequential_scheme : str, optional
        Default sequential color ramp used by heatmaps, density plots, and
        other continuous-color charts when no explicit ``cmap`` is given.
        Must be one of the recognized sequential/diverging names. Defaults
        to ``"cool_blue"`` (Paper Ink).
    diverging_scheme : str, optional
        Default diverging color ramp used by correlation matrices and other
        diverging-color charts when no explicit ``cmap`` is given. Must be
        one of the recognized sequential/diverging names. Defaults to
        ``"blue_to_red"`` (Paper Ink).
    legend_orient, legend_direction, legend_title_font_size : optional
        Legend layout.
    padding, axis_title_padding, column_padding, row_padding : optional
        Spacing in pixels.
    reference_line_color, reference_line_dash : optional
        Color and dash pattern for reference / annotation guide lines.
    strip_background_color : optional
        Facet strip-title background color.
    strip_text_size : float, optional
        Font size for facet strip titles (default 12).
    strip_padding : float, optional
        Vertical padding around facet strip titles (default 6).
    cull_threshold : int, optional
        Minimum number of pixels between adjacent axis tick labels before
        culling kicks in. Labels are dropped until the remaining labels are at
        least this many pixels apart. Default 8 (set in Rust). Set to ``0`` to
        disable culling entirely.

    Fallbacks
    ---------
    When a ``title_*`` / ``label_*`` key is unset, it falls back to its
    body-text counterpart at render time. The pair list below is generated
    from ``Theme._FALLBACKS`` (the single source of truth):

    %(fallbacks)s

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

    # Derived from the Rust ``ThemeOverridesSpec`` key manifest (the single
    # source of truth) so the Python and Rust accepted-key sets cannot drift.
    # Includes the per-level grid keys (``major_grid_*`` / ``minor_grid_*`` /
    # ``minor``) and the ``background`` alias for ``background_color``.
    _KNOWN_KEYS: frozenset[str] = frozenset(_theme_known_keys())

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

    # Every ``title_*`` / ``label_*`` key that has a body-text counterpart
    # falls back to that counterpart when unset (resolved in ``to_spec_dict``).
    # This is the single source for the fallback contract; the ``Theme``
    # docstring's pair list is derived from this table at import time.  Keys
    # with no body-text counterpart (``title_anchor``, ``title_offset``) are
    # absent; there is no public ``label_font_weight`` / ``label_font_size``
    # key, so no fallback exists for those.  ``font_size`` is the body/label
    # font-size key, so ``title_font_size`` falls back to it.
    _FALLBACKS: dict[str, str] = {
        "title_color": "font_color",
        "label_color": "font_color",
        "title_font_family": "font_family",
        "label_font_family": "font_family",
        "title_font_weight": "font_weight",
        "title_font_size": "font_size",
    }

    def to_spec_dict(self) -> dict:
        """Return a dict suitable for ``ferrum._core.render_svg(theme=...)``.

        Resolves spec-defined fallbacks (e.g. ``title_color`` falls back to
        ``font_color`` if unset) and normalises the public Python alias
        ``background`` to the Rust binding's canonical ``background_color``
        key.  CSS shorthand hex colors (``#rgb`` / ``#rgba``) are expanded by
        the Rust color parser (``from_hex_str``); Python forwards color
        strings unchanged.  Rust sees a fully-resolved dict; no Option
        fallback chains in the binding.
        """
        d = dict(self._props)
        # Expand Grid value objects: if the "grid" prop is a value object
        # (has .to_spec_dict()), call it and splice the per-level keys into
        # the dict.  Bool values (existing behavior) pass through as-is.
        if "grid" in d and hasattr(d["grid"], "to_spec_dict"):
            grid_spec = d.pop("grid").to_spec_dict()
            d.update(grid_spec)
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


# Render the docstring's fallback pair list from the single-source ``_FALLBACKS``
# table so the prose and the runtime contract cannot drift.
Theme.__doc__ = (Theme.__doc__ or "") % {
    "fallbacks": "\n    ".join(
        f"- ``{derived}`` → ``{source}``" for derived, source in Theme._FALLBACKS.items()
    )
}
