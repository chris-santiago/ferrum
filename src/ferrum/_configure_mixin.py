"""ConfigureMixin — shared configure_* methods for Chart and composition wrappers.

Both ``Chart`` (chart.py) and ``_ChartLike`` (composition.py) expose identical
``configure_*`` / ``configure`` signatures.  This mixin centralises them so a
parameter change only needs to happen in one place.

Subclasses must implement ``_append_configure(config)`` which clones self,
attaches the ``Configure`` object to the appropriate internal list, and returns
the new instance.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod

# Sentinel for "caller did not pass this kwarg" — used by the deprecated-alias
# resolver so that an explicit alias=None is still detected as "supplied".
_MISSING = object()


def _resolve_band_alias(
    canonical_value: "float | None",
    alias_value: object,
    *,
    canonical_name: str,
    alias_name: str,
    owner: str = "configure_axis",
    stacklevel: int = 3,
) -> "float | None":
    """Resolve a renamed band kwarg against its deprecated alias.

    Shared by all three surfaces that accept the ``min_band``/``max_band`` band
    kwargs with deprecated ``min_extent``/``max_extent`` aliases: the
    ``configure_axis`` mixin method, :class:`ferrum.axis.Axis`, and
    :class:`ferrum.configure.AxisConfig`.  Single-sourcing the resolution here
    keeps the three siblings from drifting on the explicit-``None`` edge (an
    explicit ``min_extent=None`` is treated as "supplied" via the
    :data:`_MISSING` sentinel, consistently across all three).

    Parameters
    ----------
    canonical_value
        Value supplied under the canonical name (``min_band`` or ``max_band``),
        or ``None`` if the caller used the default.
    alias_value
        Value supplied under the deprecated alias name (``min_extent`` or
        ``max_extent``), or :data:`_MISSING` if not supplied.
    canonical_name
        The new canonical kwarg name shown in warnings/errors.
    alias_name
        The deprecated kwarg name shown in warnings/errors.
    owner
        The owning surface name shown as the prefix of the DeprecationWarning
        (e.g. ``"Axis"``, ``"AxisConfig"``, ``"configure_axis"``).  Defaults to
        ``"configure_axis"`` for backward compatibility with the mixin call site.
    stacklevel
        The :func:`warnings.warn` ``stacklevel``.  Direct constructors
        (``Axis``/``AxisConfig``) pass ``2`` (one frame to the caller); the
        mixin method passes ``3`` (caller → ``configure_axis`` → helper).

    Returns
    -------
    float | None
        The resolved value (canonical wins; alias accepted when canonical is
        absent; both raises ``TypeError``; neither returns ``None``).

    Raises
    ------
    TypeError
        When both the canonical and alias kwarg are supplied.
    """
    if alias_value is _MISSING:
        return canonical_value
    # alias was supplied
    if canonical_value is not None:
        raise TypeError(
            f"Cannot supply both '{canonical_name}=' and the deprecated '{alias_name}=' alias; "
            f"use '{canonical_name}=' only."
        )
    warnings.warn(
        f"{owner}: '{alias_name}=' is deprecated; use '{canonical_name}=' instead. "
        "'extent' is reserved for data-domain bounds.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return alias_value  # type: ignore[return-value]


class ConfigureMixin:
    """Mixin that provides configure_axis / configure_legend / … / configure.

    Concrete classes must implement ``_append_configure`` to handle the
    clone-and-append step in the way appropriate to their internal storage.
    """

    @abstractmethod
    def _append_configure(self, config):
        """Clone self, append *config* to the internal configure list, return new instance."""

    # ------------------------------------------------------------------
    # configure_axis
    # ------------------------------------------------------------------

    def configure_axis(
        self,
        *,
        x: bool = True,
        y: bool = True,
        label_angle: "float | None" = None,
        label_font_size: "float | None" = None,
        label_color: "str | None" = None,
        label_format: "str | None" = None,
        label_format_raw: "str | None" = None,
        label_overlap: "str | None" = None,
        tick_count: "int | None" = None,
        tick_size: "float | None" = None,
        tick_values: "list | None" = None,
        title_font_size: "float | None" = None,
        title_color: "str | None" = None,
        title_padding: "float | None" = None,
        label_padding: "float | None" = None,
        domain: "bool | None" = None,
        domain_color: "str | None" = None,
        domain_width: "float | None" = None,
        grid: "bool | None" = None,
        grid_color: "str | None" = None,
        grid_dash: "list[float] | None" = None,
        grid_width: "float | None" = None,
        grid_opacity: "float | None" = None,
        domain_min: "float | None" = None,
        domain_max: "float | None" = None,
        nice: "bool | None" = None,
        zero: "bool | None" = None,
        translate: "float | None" = None,
        min_band: "float | None" = None,
        max_band: "float | None" = None,
        tick_extra: "bool | None" = None,
        tick_min_step: "float | None" = None,
        title_orient: "str | None" = None,
        zindex: "int | None" = None,
        # Deprecated aliases — accepted with a DeprecationWarning.
        min_extent: object = _MISSING,
        max_extent: object = _MISSING,
    ):
        """Apply axis configuration.

        Parameters
        ----------
        x, y : bool, default True
            Deprecated and has no effect. Use ``Chart.axis(x=False)`` /
            ``Chart.axis(y=False)`` to show or hide an axis.
        label_angle : float, optional
            Tick label rotation in degrees.
        label_font_size : float, optional
            Tick label font size in pixels.
        label_color : str, optional
            Tick label color.
        label_format : str, optional
            Named format preset (e.g. ``"currency"``, ``"percent"``).
        label_format_raw : str, optional
            Raw d3-format string (e.g. ``".2f"``, ``"~s"``); overrides
            ``label_format`` when both are given.
        label_overlap : str, optional
            Overlap-handling strategy for crowded tick labels
            (e.g. ``"parity"``, ``"greedy"``).
        tick_count : int, optional
            Target number of ticks on the axis.
        tick_size : float, optional
            Tick mark length in pixels.
        tick_values : list, optional
            Explicit tick positions, overriding automatic tick selection.
        title_font_size : float, optional
            Axis title font size in pixels.
        title_color : str, optional
            Axis title color.
        title_padding : float, optional
            Gap between the axis title and its labels in pixels.
        label_padding : float, optional
            Gap between tick labels and the axis line in pixels.
        domain : bool, optional
            Whether to draw the axis domain (baseline) line.
        domain_color : str, optional
            Domain line color.
        domain_width : float, optional
            Domain line width in pixels.
        grid : bool, optional
            Whether to draw grid lines for this axis.
        grid_color : str, optional
            Grid line color.
        grid_dash : list[float], optional
            Grid line dash pattern (on/off pixel lengths).
        grid_width : float, optional
            Grid line width in pixels.
        grid_opacity : float, optional
            Grid line opacity (0--1).
        domain_min, domain_max : float, optional
            Explicit scale domain bounds.
        nice : bool, optional
            Extend the domain to nice round numbers.
        zero : bool, optional
            Include zero in the scale domain.
        translate : float, optional
            Pixel translation of the axis group perpendicular to its line.
        min_band, max_band : float, optional
            Lower / upper bounds for the reserved axis margin band, in pixels.

            .. deprecated:: 0.17.0
                ``min_extent=`` and ``max_extent=`` are accepted as aliases
                but are deprecated; use ``min_band=`` / ``max_band=`` instead.
        tick_extra : bool, optional
            Append an extra tick at each domain boundary.
        tick_min_step : float, optional
            Minimum step between ticks in data space.
        title_orient : str, optional
            Side/orientation of the axis title.
        zindex : int, optional
            Coarse draw order of the axis relative to marks.

        Returns
        -------
        Self

        Notes
        -----
        ``orient`` (the axis side) is intentionally absent here: ``configure_axis``
        applies to both the x and y axes, so a single side value can be valid for
        at most one of them.  Set it per-axis instead, via
        ``configure(axis_x=AxisConfig(orient="top"))`` /
        ``configure(axis_y=AxisConfig(orient="right"))`` or per-channel
        ``fm.X("f", axis=fm.Axis(orient="top"))``.
        """
        from ferrum.configure import AxisConfig, Configure

        # Resolve deprecated aliases before forwarding to AxisConfig.
        min_band = _resolve_band_alias(
            min_band, min_extent, canonical_name="min_band", alias_name="min_extent"
        )
        max_band = _resolve_band_alias(
            max_band, max_extent, canonical_name="max_band", alias_name="max_extent"
        )

        cfg = AxisConfig(
            x=x,
            y=y,
            label_angle=label_angle,
            label_font_size=label_font_size,
            label_color=label_color,
            label_format=label_format,
            label_format_raw=label_format_raw,
            label_overlap=label_overlap,
            tick_count=tick_count,
            tick_size=tick_size,
            tick_values=tick_values,
            title_font_size=title_font_size,
            title_color=title_color,
            title_padding=title_padding,
            label_padding=label_padding,
            domain=domain,
            domain_color=domain_color,
            domain_width=domain_width,
            grid=grid,
            grid_color=grid_color,
            grid_dash=grid_dash,
            grid_width=grid_width,
            grid_opacity=grid_opacity,
            domain_min=domain_min,
            domain_max=domain_max,
            nice=nice,
            zero=zero,
            translate=translate,
            min_band=min_band,
            max_band=max_band,
            tick_extra=tick_extra,
            tick_min_step=tick_min_step,
            title_orient=title_orient,
            zindex=zindex,
        )
        return self._append_configure(Configure(axis=cfg))

    # ------------------------------------------------------------------
    # configure_legend
    # ------------------------------------------------------------------

    def configure_legend(
        self,
        *,
        orient: "str | None" = None,
        direction: "str | None" = None,
        columns: "int | None" = None,
        title_font_size: "float | None" = None,
        label_font_size: "float | None" = None,
        label_color: "str | None" = None,
        label_limit: "float | None" = None,
        symbol_size: "float | None" = None,
        symbol_stroke_width: "float | None" = None,
        symbol_type: "str | None" = None,
        gradient_length: "float | None" = None,
        gradient_thickness: "float | None" = None,
        title_padding: "float | None" = None,
        row_padding: "float | None" = None,
        column_padding: "float | None" = None,
        clip_height: "float | None" = None,
        tick_min_step: "float | None" = None,
        offset: "float | None" = None,
        padding: "float | None" = None,
        zindex: "int | None" = None,
    ):
        """Apply legend configuration.

        Parameters
        ----------
        orient : str, optional
            Legend position: ``"right"``, ``"left"``, ``"top"``, ``"bottom"``, ``"none"``.
        direction : str, optional
            Layout direction: ``"vertical"`` or ``"horizontal"``.
        columns : int, optional
            Number of columns for multi-column layout.
        title_font_size : float, optional
            Legend title font size in pixels.
        label_font_size : float, optional
            Legend entry label font size in pixels.
        label_color : str, optional
            Legend label color.
        label_limit : float, optional
            Maximum legend-label width in pixels; wider labels are truncated
            with an ellipsis.
        symbol_size : float, optional
            Legend symbol size in pixels.
        symbol_stroke_width : float, optional
            Stroke width of legend symbol swatches in pixels.
        symbol_type : str, optional
            Legend symbol shape (e.g. ``"circle"``, ``"square"``).
        gradient_length : float, optional
            Length of the continuous-legend gradient bar in pixels.
        gradient_thickness : float, optional
            Thickness of the continuous-legend gradient bar in pixels.
        title_padding : float, optional
            Padding between the legend title and its entries in pixels.
        row_padding : float, optional
            Vertical entry spacing in pixels (vertical-direction legends).
        column_padding : float, optional
            Horizontal entry spacing in pixels (horizontal-direction legends).
        clip_height : float, optional
            Cap on the legend group height in pixels; overflow is hard-clipped.
        tick_min_step : float, optional
            Minimum step between colorbar ticks in data units.
        offset : float, optional
            Legend offset from the plot edge in pixels.
        padding : float, optional
            Padding inside the legend box in pixels.
        zindex : int, optional
            Coarse draw order of the legend relative to marks.

        Returns
        -------
        Self
        """
        from ferrum.configure import LegendConfig, Configure

        cfg = LegendConfig(
            orient=orient,
            direction=direction,
            columns=columns,
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            label_color=label_color,
            label_limit=label_limit,
            symbol_size=symbol_size,
            symbol_stroke_width=symbol_stroke_width,
            symbol_type=symbol_type,
            gradient_length=gradient_length,
            gradient_thickness=gradient_thickness,
            title_padding=title_padding,
            row_padding=row_padding,
            column_padding=column_padding,
            clip_height=clip_height,
            tick_min_step=tick_min_step,
            offset=offset,
            padding=padding,
            zindex=zindex,
        )
        return self._append_configure(Configure(legend=cfg))

    # ------------------------------------------------------------------
    # configure_title
    # ------------------------------------------------------------------

    def configure_title(
        self,
        *,
        font_size: "float | None" = None,
        font_weight: "str | None" = None,
        anchor: "str | None" = None,
        color: "str | None" = None,
        offset: "float | None" = None,
        subtitle_font_size: "float | None" = None,
        subtitle_color: "str | None" = None,
    ):
        """Apply title configuration.

        Parameters
        ----------
        font_size : float, optional
            Title font size in pixels.
        font_weight : str, optional
            Title font weight (e.g. ``"normal"``, ``"bold"``).
        anchor : str, optional
            Title alignment: ``"start"``, ``"middle"``, or ``"end"``.
        color : str, optional
            Title text color.
        offset : float, optional
            Title offset from the plot in pixels.
        subtitle_font_size : float, optional
            Subtitle font size in pixels.
        subtitle_color : str, optional
            Subtitle text color.

        Returns
        -------
        Self
        """
        from ferrum.configure import TitleConfig, Configure

        cfg = TitleConfig(
            font_size=font_size,
            font_weight=font_weight,
            anchor=anchor,
            color=color,
            offset=offset,
            subtitle_font_size=subtitle_font_size,
            subtitle_color=subtitle_color,
        )
        return self._append_configure(Configure(title=cfg))

    # ------------------------------------------------------------------
    # configure_grid
    # ------------------------------------------------------------------

    def configure_grid(
        self,
        *,
        x: "bool | None" = None,
        y: "bool | None" = None,
        color: "str | None" = None,
        width: "float | None" = None,
        dash: "list[float] | None" = None,
        opacity: "float | None" = None,
        band_colors: "list[str] | None" = None,
    ):
        """Apply grid configuration.

        Parameters
        ----------
        x, y : bool, optional
            Enable/disable grid on each axis.
        color : str, optional
            Grid line color.
        width : float, optional
            Grid line width in pixels.
        dash : list[float], optional
            Grid line dash pattern (on/off pixel lengths).
        opacity : float, optional
            Grid line opacity (0--1).
        band_colors : list[str], optional
            Alternating band fill colors (``None`` to disable).

        Returns
        -------
        Self
        """
        from ferrum.configure import GridConfig, Configure

        cfg = GridConfig(
            x=x,
            y=y,
            color=color,
            width=width,
            dash=dash,
            opacity=opacity,
            band_colors=band_colors,
        )
        return self._append_configure(Configure(grid=cfg))

    # ------------------------------------------------------------------
    # configure_padding
    # ------------------------------------------------------------------

    def configure_padding(
        self,
        *,
        top: "float | None" = None,
        right: "float | None" = None,
        bottom: "float | None" = None,
        left: "float | None" = None,
        auto: bool = True,
    ):
        """Apply padding configuration.

        Parameters
        ----------
        top, right, bottom, left : float, optional
            Minimum padding in pixels per side.
        auto : bool, default True
            Auto-expand margins to fit labels.

        Returns
        -------
        Self
        """
        from ferrum.configure import PaddingConfig, Configure

        cfg = PaddingConfig(top=top, right=right, bottom=bottom, left=left, auto=auto)
        return self._append_configure(Configure(padding=cfg))

    # ------------------------------------------------------------------
    # configure_color
    # ------------------------------------------------------------------

    def configure_color(
        self,
        *,
        scheme: "str | None" = None,
        sequential_scheme: "str | None" = None,
        diverging_scheme: "str | None" = None,
        domain: "list | None" = None,
        range: "list[str] | None" = None,
    ):
        """Apply color scale configuration.

        Parameters
        ----------
        scheme : str, optional
            Categorical color scheme name.
        sequential_scheme : str, optional
            Sequential color scheme name.
        diverging_scheme : str, optional
            Diverging color scheme name.
        domain : list, optional
            Explicit color-scale domain values.
        range : list[str], optional
            Explicit color range, overriding the scheme.

        Returns
        -------
        Self
        """
        from ferrum.configure import ColorConfig, Configure

        cfg = ColorConfig(
            scheme=scheme,
            sequential_scheme=sequential_scheme,
            diverging_scheme=diverging_scheme,
            domain=domain,
            range=range,
        )
        return self._append_configure(Configure(color=cfg))

    # ------------------------------------------------------------------
    # configure  (unified method — accepts typed config objects directly)
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        axis: "AxisConfig | None" = None,
        axis_x: "AxisConfig | None" = None,
        axis_y: "AxisConfig | None" = None,
        axis_y2: "AxisConfig | None" = None,
        legend: "LegendConfig | None" = None,
        title: "TitleConfig | None" = None,
        grid: "GridConfig | None" = None,
        padding: "PaddingConfig | None" = None,
        color: "ColorConfig | None" = None,
    ):
        """Append a [Configure][ferrum.Configure] layer.

        Accepts typed config objects for each domain.  All parameters are
        keyword-only and default to ``None`` (no change for that domain).

        Parameters
        ----------
        axis : AxisConfig, optional
            Applies to all axes.
        axis_x : AxisConfig, optional
            Applies only to the x axis.
        axis_y : AxisConfig, optional
            Applies only to the y axis.
        axis_y2 : AxisConfig, optional
            Applies only to the secondary y axis.
        legend : LegendConfig, optional
            Legend appearance.
        title : TitleConfig, optional
            Chart title appearance.
        grid : GridConfig, optional
            Grid line appearance.
        padding : PaddingConfig, optional
            Plot-area padding.
        color : ColorConfig, optional
            Default color scale settings.

        Returns
        -------
        Self
            New instance with the configure layer appended.

        Examples
        --------
        >>> from ferrum.configure import AxisConfig, LegendConfig
        >>> chart.configure(
        ...     axis=AxisConfig(label_angle=-45),
        ...     legend=LegendConfig(orient="bottom"),
        ... )
        """
        from ferrum.configure import Configure

        cfg = Configure(
            axis=axis,
            axis_x=axis_x,
            axis_y=axis_y,
            axis_y2=axis_y2,
            legend=legend,
            title=title,
            grid=grid,
            padding=padding,
            color=color,
        )
        return self._append_configure(cfg)
