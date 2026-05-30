"""Grid theme value class — spec §3.19."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Grid:
    """Theme-level grid configuration controlling major and minor gridlines.

    Accepted by ``Theme(grid=...)`` / ``Theme.update(grid=fr.Grid(...))``.
    Passing a bare ``bool`` remains valid (maps to the ``major`` enable flag).

    **Shorthand resolution.** The bare ``color``, ``width``, ``dash``, and
    ``opacity`` parameters are a *fallback that sets both levels*.  An explicit
    ``major_*`` or ``minor_*`` value for that property overrides the bare value
    for that level only.  Examples::

        Grid(color="#f5f5f5")
        # → major_grid_color="#f5f5f5", minor_grid_color="#f5f5f5"

        Grid(color="#eee", minor_color="#f8f8f8")
        # → major_grid_color="#eee", minor_grid_color="#f8f8f8"

    The bare shorthand keys (``color``, ``width``, ``dash``, ``opacity``) are
    **never emitted** to the Rust renderer; only resolved per-level keys appear
    in ``to_spec_dict()`` output.

    **Minor gridlines.** ``minor=True`` renders minor gridlines on continuous
    axes (linear, log, time, pow, sqrt, symlog).  On categorical or
    discretising axes (band, ordinal, quantile, threshold, bin-ordinal) there
    is no continuum to subdivide, so ``minor=True`` there is a semantic
    absence — no extra lines are drawn and no error is raised.

    Parameters
    ----------
    major : bool, default True
        Enable major gridlines.
    minor : bool, default False
        Enable minor gridlines (continuous scales only).
    color : str, optional
        Fallback color for both major and minor gridlines.
    width : float, optional
        Fallback stroke width for both levels.
    dash : list of float, optional
        Fallback dash pattern (SVG stroke-dasharray lengths) for both levels.
    opacity : float, optional
        Fallback stroke opacity for both levels.
    major_color : str, optional
        Major gridline color; overrides bare ``color`` for the major level.
    minor_color : str, optional
        Minor gridline color; overrides bare ``color`` for the minor level.
    major_width : float, optional
        Major gridline stroke width; overrides bare ``width`` for major.
    minor_width : float, optional
        Minor gridline stroke width; overrides bare ``width`` for minor.
    major_dash : list of float, optional
        Major gridline dash pattern; overrides bare ``dash`` for major.
    minor_dash : list of float, optional
        Minor gridline dash pattern; overrides bare ``dash`` for minor.
    major_opacity : float, optional
        Major gridline opacity; overrides bare ``opacity`` for major.
    minor_opacity : float, optional
        Minor gridline opacity; overrides bare ``opacity`` for minor.

    Examples
    --------
    >>> import ferrum as fm
    >>> # Enable minor gridlines, both levels same colour
    >>> g = fm.Grid(minor=True, color="#f5f5f5")
    >>> g.to_spec_dict()
    {'grid': True, 'minor': True, 'major_grid_color': '#f5f5f5', 'minor_grid_color': '#f5f5f5'}

    >>> # Different colours per level
    >>> g2 = fm.Grid(minor=True, color="#eeeeee", minor_color="#f8f8f8")
    >>> g2.to_spec_dict()
    {'grid': True, 'minor': True, 'major_grid_color': '#eeeeee', 'minor_grid_color': '#f8f8f8'}

    >>> # Attach to a theme
    >>> chart.theme(fm.themes.minimal.update(grid=fm.Grid(minor=True)))
    """

    major: bool = True
    minor: bool = False
    # Bare shorthand — fallback that sets both levels.
    color: Optional[str] = None
    width: Optional[float] = None
    dash: Optional[list] = None
    opacity: Optional[float] = None
    # Per-level overrides.
    major_color: Optional[str] = None
    minor_color: Optional[str] = None
    major_width: Optional[float] = None
    minor_width: Optional[float] = None
    major_dash: Optional[list] = None
    minor_dash: Optional[list] = None
    major_opacity: Optional[float] = None
    minor_opacity: Optional[float] = None

    def to_spec_dict(self) -> dict:
        """Serialize to the dict shape the Rust binding expects.

        Always emits ``grid`` (major enable) and ``minor`` (minor enable).
        Resolves shorthand → per-level: for each of color/width/dash/opacity,
        major level = ``major_X`` if not ``None`` else bare ``X``; minor level =
        ``minor_X`` if not ``None`` else bare ``X``.  Only non-``None`` resolved
        per-level styling keys are emitted.  The bare shorthand keys are
        **never** present in the output.
        """
        out: dict = {
            "grid": self.major,
            "minor": self.minor,
        }
        # color
        resolved_major_color = self.major_color if self.major_color is not None else self.color
        resolved_minor_color = self.minor_color if self.minor_color is not None else self.color
        if resolved_major_color is not None:
            out["major_grid_color"] = resolved_major_color
        if resolved_minor_color is not None:
            out["minor_grid_color"] = resolved_minor_color

        # width
        resolved_major_width = self.major_width if self.major_width is not None else self.width
        resolved_minor_width = self.minor_width if self.minor_width is not None else self.width
        if resolved_major_width is not None:
            out["major_grid_width"] = resolved_major_width
        if resolved_minor_width is not None:
            out["minor_grid_width"] = resolved_minor_width

        # dash
        resolved_major_dash = self.major_dash if self.major_dash is not None else self.dash
        resolved_minor_dash = self.minor_dash if self.minor_dash is not None else self.dash
        if resolved_major_dash is not None:
            out["major_grid_dash"] = resolved_major_dash
        if resolved_minor_dash is not None:
            out["minor_grid_dash"] = resolved_minor_dash

        # opacity
        resolved_major_opacity = (
            self.major_opacity if self.major_opacity is not None else self.opacity
        )
        resolved_minor_opacity = (
            self.minor_opacity if self.minor_opacity is not None else self.opacity
        )
        if resolved_major_opacity is not None:
            out["major_grid_opacity"] = resolved_major_opacity
        if resolved_minor_opacity is not None:
            out["minor_grid_opacity"] = resolved_minor_opacity

        return out
