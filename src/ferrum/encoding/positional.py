"""Positional encoding channels (X, Y, X2, Y2, errors, polar).

Each channel references one named role from :mod:`ferrum.encoding._honored`;
``_honored_kwargs`` is the single, machine-readable source of truth for which
kwargs the channel honors (see the ``ChannelBase`` docstring for the contract).
"""

from __future__ import annotations

from ferrum.encoding._honored import (
    PRIMARY_POSITIONAL,
    SECONDARY_EXTENT,
    POLAR_PRIMARY,
)
from ferrum.encoding.base import ChannelBase


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("hp", type_="Q"))
    >>> fm.Chart(df).encode(x=fm.X("hp", bin=True))
    >>> fm.Chart(df).encode(x=fm.X("hp", aggregate="mean"))
    """

    _channel_name = "x"
    _honored_kwargs = PRIMARY_POSITIONAL
    _stack_kwarg = True


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(y=fm.Y("mpg", type_="Q"))
    >>> fm.Chart(df).encode(y=fm.Y("mpg", aggregate="mean"))
    """

    _channel_name = "y"
    _honored_kwargs = PRIMARY_POSITIONAL
    _stack_kwarg = True


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("start"), x2=fm.X2("end"))
    """

    _channel_name = "x2"

    _honored_kwargs = SECONDARY_EXTENT


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(y=fm.Y("low"), y2=fm.Y2("high"))
    """

    _channel_name = "y2"

    _honored_kwargs = SECONDARY_EXTENT


class XError(ChannelBase):
    """X-axis error channel — maps a field to symmetric error around x.

    The error bar extends ``x ± x_error`` along the horizontal axis.

    Parameters
    ----------
    field : str
        Column name whose values are the error magnitude.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="mean_hp", x_error=fm.XError("ci_hp"))
    """

    _channel_name = "x_error"

    _honored_kwargs = SECONDARY_EXTENT


class YError(ChannelBase):
    """Y-axis error channel — maps a field to symmetric error around y.

    The error bar extends ``y ± y_error`` along the vertical axis.

    Parameters
    ----------
    field : str
        Column name whose values are the error magnitude.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(y="mean_mpg", y_error=fm.YError("ci_mpg"))
    """

    _channel_name = "y_error"

    _honored_kwargs = SECONDARY_EXTENT


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

    _honored_kwargs = SECONDARY_EXTENT


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

    _honored_kwargs = SECONDARY_EXTENT


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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(theta=fm.Theta("count", stack=True))
    """

    _channel_name = "theta"

    _honored_kwargs = POLAR_PRIMARY
    _stack_kwarg = True


class Radius(ChannelBase):
    """Polar radius channel — maps a field to the radial position in polar coords.

    Controls how far each mark is placed from the center of the polar plot.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.
    stack : bool or str, optional
        Stacking behaviour for radial bars (wind-rose / coxcomb charts).
        ``True`` or ``"zero"`` accumulates segments outward from the origin;
        ``"normalize"`` produces percentage rings; ``"center"`` produces a
        symmetric streamgraph-style layout.  ``False`` or ``None`` disables
        stacking.  Requires ``mark_bar`` under ``CoordPolar``.

    Notes
    -----
    Requires ``CoordPolar()`` on the chart to activate polar rendering;
    without it the channel is registered but the mark renders in Cartesian
    space.

    When ``stack`` is set, the radius channel is remapped to ``y`` (the value
    axis) before reaching Rust's ``apply_stack``; ``bar.rs build_polar`` then
    reads ``__stack_y_base__`` as the inner radius so each segment starts
    where the previous one ended.  Pass ``position=fm.Stack()`` on
    ``mark_bar()`` as an alternative.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(theta=fm.Theta("count"), radius=fm.Radius("distance"))
    >>> # Stacked wind-rose / coxcomb:
    >>> fm.Chart(df).encode(
    ...     theta=fm.Theta("direction"),
    ...     radius=fm.Radius("speed", stack=True),
    ...     color="category:N",
    ... ).coord(fm.CoordPolar(theta="x"))
    """

    _channel_name = "radius"

    _honored_kwargs = POLAR_PRIMARY
    _stack_kwarg = True


class Theta2(ChannelBase):
    """Polar second-extent angle channel — maps a field to the angular end position.

    Used with ``Theta`` to define an explicit angular span for each arc segment
    (annular wedge marks).  When both ``theta`` and ``theta2`` are encoded, the
    arc sweeps from the ``theta`` value to the ``theta2`` value directly from
    data, rather than computing a proportional sweep from column sums.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Requires ``CoordPolar()`` on the chart to activate polar rendering.
    Remapped to ``x2`` (when ``CoordPolar(theta="x")``) or ``y2``
    (when ``CoordPolar(theta="y")``) before the spec reaches Rust.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(
    ...     theta=fm.Theta("t_start"),
    ...     theta2=fm.Theta2("t_end"),
    ...     radius=fm.Radius("r_inner"),
    ...     radius2=fm.Radius2("r_outer"),
    ... )
    """

    _channel_name = "theta2"

    _honored_kwargs = SECONDARY_EXTENT


class Radius2(ChannelBase):
    """Polar second-extent radius channel — maps a field to the outer radial boundary.

    Used with ``Radius`` to define an explicit inner/outer radial span for each
    arc segment (annular wedge marks).  When both ``radius`` and ``radius2`` are
    encoded, the arc spans from the ``radius`` column value (inner boundary) to
    the ``radius2`` column value (outer boundary) per row.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from the column dtype when omitted.

    Notes
    -----
    Requires ``CoordPolar()`` on the chart to activate polar rendering.
    Remapped to ``y2`` (when ``CoordPolar(theta="x")``) or ``x2``
    (when ``CoordPolar(theta="y")``) before the spec reaches Rust.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(
    ...     theta=fm.Theta("t_start"),
    ...     theta2=fm.Theta2("t_end"),
    ...     radius=fm.Radius("r_inner"),
    ...     radius2=fm.Radius2("r_outer"),
    ... )
    """

    _channel_name = "radius2"

    _honored_kwargs = SECONDARY_EXTENT
