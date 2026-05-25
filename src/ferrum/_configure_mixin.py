"""ConfigureMixin — shared configure_* methods for Chart and composition wrappers.

Both ``Chart`` (chart.py) and ``_ChartLike`` (composition.py) expose identical
``configure_*`` / ``configure`` signatures.  This mixin centralises them so a
parameter change only needs to happen in one place.

Subclasses must implement ``_append_configure(config)`` which clones self,
attaches the ``Configure`` object to the appropriate internal list, and returns
the new instance.
"""

from __future__ import annotations

from abc import abstractmethod


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
        domain_min: "float | None" = None,
        domain_max: "float | None" = None,
        nice: "bool | None" = None,
        zero: "bool | None" = None,
    ):
        """Apply axis configuration.

        Parameters
        ----------
        x, y : bool, default True
            Which axes this config applies to.
        label_angle : float, optional
            Tick label rotation in degrees.
        label_format : str, optional
            Named format preset (e.g. ``"currency"``, ``"percent"``).
        domain_min, domain_max : float, optional
            Explicit scale domain bounds.

        Returns
        -------
        Self
        """
        from ferrum.configure import AxisConfig, Configure

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
            domain_min=domain_min,
            domain_max=domain_max,
            nice=nice,
            zero=zero,
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
        symbol_size: "float | None" = None,
        symbol_type: "str | None" = None,
        gradient_length: "float | None" = None,
        offset: "float | None" = None,
        padding: "float | None" = None,
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
            symbol_size=symbol_size,
            symbol_type=symbol_type,
            gradient_length=gradient_length,
            offset=offset,
            padding=padding,
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
            Title font size.
        anchor : str, optional
            Title alignment: ``"start"``, ``"middle"``, or ``"end"``.

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
        """Append a :class:`~ferrum.configure.Configure` layer.

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
