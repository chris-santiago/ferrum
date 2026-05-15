"""Positional encoding channels (X, Y, X2, Y2, errors, polar)."""

from __future__ import annotations

from ferrum.encoding.base import ChannelBase


_RENDERED_HONORED = frozenset([
    "type", "bin", "aggregate", "scale", "title",
    # Sort — honored by scale_resolve.rs ordinal domain builder.
    "sort",
    # Axis dict — honored by prepare.rs AxisInput construction.
    "axis",
    # Stack — honored by position.rs Stack strategy selection.
    "stack",
    # Impute dict — honored by prepare.rs apply_impute.
    "impute",
    # Format string and type — honored by prepare.rs apply_tick_format.
    "format", "format_type",
    # Legend dict — honored by prepare.rs legend_orient_override / title.
    "legend",
])


class X(ChannelBase):
    """Positional X channel — maps a field to the horizontal axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type: quantitative, nominal, ordinal, temporal. Inferred from
        the column dtype when omitted.
    bin : bool or Bin, optional
        If truthy, bin the field before mapping.  Pass a ``Bin(...)`` instance
        to control bin width or count; ``True`` uses automatic binning.
    aggregate : str, optional
        Aggregation operation applied before mapping (e.g. ``"mean"``,
        ``"sum"``, ``"count"``).
    scale : Scale, optional
        Explicit scale override (e.g. ``LogScale()``, ``LinearScale()``).
    title : str, optional
        Axis title override.  When omitted the field name is used.

    Notes
    -----
    ``axis``, ``legend``, ``sort``, ``stack``, and ``impute`` kwargs are
    accepted and forwarded to the EncodingSpec; per-channel axis/legend
    customization depends on Rust-side support for the channel.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("hp", type_="Q"))
    >>> fm.Chart(df).encode(x=fm.X("hp", bin=True))
    >>> fm.Chart(df).encode(x=fm.X("hp", aggregate="mean"))
    """

    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED


class Y(ChannelBase):
    """Positional Y channel — maps a field to the vertical axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type: quantitative, nominal, ordinal, temporal. Inferred from
        the column dtype when omitted.
    bin : bool or Bin, optional
        If truthy, bin the field before mapping.  Pass a ``Bin(...)`` instance
        to control bin width or count; ``True`` uses automatic binning.
    aggregate : str, optional
        Aggregation operation applied before mapping (e.g. ``"mean"``,
        ``"sum"``, ``"count"``).
    scale : Scale, optional
        Explicit scale override (e.g. ``LogScale()``, ``LinearScale()``).
    title : str, optional
        Axis title override.  When omitted the field name is used.

    Notes
    -----
    ``axis``, ``legend``, ``sort``, ``stack``, and ``impute`` kwargs are
    accepted and forwarded to the EncodingSpec; per-channel axis/legend
    customization depends on Rust-side support for the channel.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(y=fm.Y("mpg", type_="Q"))
    >>> fm.Chart(df).encode(y=fm.Y("mpg", aggregate="mean"))
    """

    _channel_name = "y"
    _renders_in_phase_8a = True
    _honored_kwargs = _RENDERED_HONORED

    _VALID_STACK = frozenset(("zero", "normalize", "center", "false", "null", "none"))

    def _validate(self) -> None:
        super()._validate()
        stack = self._kwargs.get("stack")
        if stack is not None and isinstance(stack, str):
            if stack.lower() not in self._VALID_STACK:
                raise ValueError(
                    f"Y(stack={stack!r}): must be one of "
                    "'zero', 'normalize', 'center', or None; "
                    f"got {stack!r}"
                )


class X2(ChannelBase):
    """Secondary X channel — maps a field to the second x position.

    Used for ranged marks (``rule``, ``rect``, ``ribbon``) where a mark spans
    from ``x`` to ``x2`` along the horizontal axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    ``bin``, ``aggregate``, ``scale``, and ``title`` kwargs are accepted but
    are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("start"), x2=fm.X2("end"))
    """

    _channel_name = "x2"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])


class Y2(ChannelBase):
    """Secondary Y channel — maps a field to the second y position.

    Used for ranged marks (``rule``, ``rect``, ``ribbon``) where a mark spans
    from ``y`` to ``y2`` along the vertical axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    ``bin``, ``aggregate``, ``scale``, and ``title`` kwargs are accepted but
    are reserved for future use (no-op today) — they trigger a one-time
    deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(y=fm.Y("low"), y2=fm.Y2("high"))
    """

    _channel_name = "y2"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])


class XError(ChannelBase):
    """X-axis error channel — maps a field to symmetric error around x.

    The error bar extends ``x ± x_error`` along the horizontal axis.

    Parameters
    ----------
    field : str
        Column name whose values are the error magnitude.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="mean_hp", x_error=fm.XError("ci_hp"))
    """

    _channel_name = "x_error"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])


class YError(ChannelBase):
    """Y-axis error channel — maps a field to symmetric error around y.

    The error bar extends ``y ± y_error`` along the vertical axis.

    Parameters
    ----------
    field : str
        Column name whose values are the error magnitude.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(y="mean_mpg", y_error=fm.YError("ci_mpg"))
    """

    _channel_name = "y_error"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])


class XError2(ChannelBase):
    """Secondary x-axis error channel — for asymmetric error bounds.

    When paired with ``XError``, sets the upper bound of the error bar
    independently of the lower bound, enabling asymmetric error bars.

    Parameters
    ----------
    field : str
        Column name whose values are the upper-side error magnitude.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(
    ...     x="mean_hp",
    ...     x_error=fm.XError("err_low"),
    ...     x_error2=fm.XError2("err_high"),
    ... )
    """

    _channel_name = "x_error2"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])


class YError2(ChannelBase):
    """Secondary y-axis error channel — for asymmetric error bounds.

    When paired with ``YError``, sets the upper bound of the error bar
    independently of the lower bound, enabling asymmetric error bars.

    Parameters
    ----------
    field : str
        Column name whose values are the upper-side error magnitude.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(
    ...     y="mean_mpg",
    ...     y_error=fm.YError("err_low"),
    ...     y_error2=fm.YError2("err_high"),
    ... )
    """

    _channel_name = "y_error2"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])


class Theta(ChannelBase):
    """Polar angle channel — maps a field to the angular position in polar coords.

    Typically used with arc or pie marks; the field values determine the sweep
    angle of each arc segment.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    stack : bool or str, optional
        Stacking behaviour for arc segments.  ``True`` enables stacking;
        ``False`` disables it; ``"normalize"`` produces percentage arcs.

    Notes
    -----
    Requires ``CoordPolar()`` on the chart to activate polar rendering;
    without it the channel is registered but the mark renders in Cartesian
    space.

    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(theta=fm.Theta("count", stack=True))
    """

    _channel_name = "theta"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "stack"])


class Radius(ChannelBase):
    """Polar radius channel — maps a field to the radial position in polar coords.

    Controls how far each mark is placed from the center of the polar plot.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Requires ``CoordPolar()`` on the chart to activate polar rendering;
    without it the channel is registered but the mark renders in Cartesian
    space.

    Other kwargs are accepted but are reserved for future use (no-op today)
    — they trigger a one-time deprecation warning.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(theta=fm.Theta("count"), radius=fm.Radius("distance"))
    """

    _channel_name = "radius"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type"])
