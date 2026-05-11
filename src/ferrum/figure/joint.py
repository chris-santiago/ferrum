"""Joint distribution convenience functions (jointplot)."""
from __future__ import annotations
from typing import Any

from ferrum import Bin2D, Chart, JointChart


_VALID_CENTER_KINDS = {"scatter", "kde", "hist", "hex", "reg"}
_VALID_MARGINAL_KINDS = {"hist", "kde", "rug", "box"}


def jointplot(
    data: Any, *,
    x: str, y: str, hue: Any = None,
    kind: str = "scatter",
    marginal_kind: str = "hist",
    ratio: int = 5, space: float = 0.05,
    xlim: Any = None, ylim: Any = None,
    joint_kws: Any = None, marginal_kws: Any = None,
    height: float | None = None, theme: Any = None,
    **encode_kwargs: Any,
) -> JointChart:
    """Joint-distribution plot with marginals.

    Builds a ``JointChart`` composed of a central bivariate plot flanked by
    univariate marginals along the ``x`` (top) and ``y`` (right) axes.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str
        Column name for the horizontal axis (required).
    y : str
        Column name for the vertical axis (required).
    hue : str or encoding, optional
        Column name to map to color in both the center and marginal charts.
    kind : {"scatter", "kde", "hist", "hex", "reg"}, default "scatter"
        Mark to use for the central panel.  ``"scatter"`` draws
        ``mark_point``; ``"kde"`` draws ``mark_density``; ``"hist"``
        draws a 2-D histogram via ``Bin2D`` + ``mark_rect``; ``"hex"``
        draws ``mark_hex``; ``"reg"`` layers ``mark_point`` + a
        ``mark_smooth(method="lm")`` fit line.
    marginal_kind : {"hist", "kde", "rug", "box"}, default "hist"
        Mark to use for the marginal panels (same kind applied to both
        the top x-marginal and the right y-marginal).
    ratio : int, default 5
        Size ratio of the center panel to the marginal panels.
    space : float, default 0.05
        Gap (in layout units) between the center and marginal panels.
    xlim : tuple, optional
        ``(min, max)`` domain for the x-axis (reserved; passed to
        ``JointChart`` for future renderer support).
    ylim : tuple, optional
        ``(min, max)`` domain for the y-axis (reserved; passed to
        ``JointChart`` for future renderer support).
    joint_kws : dict, optional
        Extra keyword arguments forwarded to the center-panel mark call.
    marginal_kws : dict, optional
        Extra keyword arguments forwarded to the marginal mark calls.
    height : float or None, optional
        Height and width of the square central panel in pixels.
    theme : Theme, optional
        Visual theme applied to all three panels via ``Chart.theme()``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()`` on
        the center chart.

    Returns
    -------
    JointChart
        Compound view with ``center``, ``top``, and ``right`` sub-charts.

    Raises
    ------
    ValueError
        If ``kind`` or ``marginal_kind`` is not one of the supported values.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.jointplot(df, x="sepal_length", y="sepal_width")

    2-D histogram center with KDE marginals, colored by species:

    >>> fm.jointplot(
    ...     df, x="sepal_length", y="petal_length",
    ...     kind="hist", marginal_kind="kde", hue="species",
    ... )
    """
    if kind not in _VALID_CENTER_KINDS:
        raise ValueError(
            f"jointplot: kind must be one of {sorted(_VALID_CENTER_KINDS)}; got {kind!r}"
        )
    if marginal_kind not in _VALID_MARGINAL_KINDS:
        raise ValueError(
            f"jointplot: marginal_kind must be one of {sorted(_VALID_MARGINAL_KINDS)}; "
            f"got {marginal_kind!r}"
        )

    # Build the center chart per `kind`.
    jk = dict(joint_kws or {})

    enc_center: dict = {"x": x, "y": y}
    if hue is not None:
        enc_center["color"] = hue
    enc_center.update(encode_kwargs)

    if kind == "scatter":
        center = Chart(data).mark_point(**jk).encode(**enc_center)
    elif kind == "kde":
        center = Chart(data).mark_density(**jk).encode(**enc_center)
    elif kind == "hist":
        # Use Bin2D + mark_rect for 2D histogram.
        bin2d_kwargs: dict = {}
        for k in ("bins_x", "bins_y"):
            if k in jk:
                bin2d_kwargs[k] = jk.pop(k)
        center = (
            Chart(data)
            .transform(Bin2D(x=x, y=y, **bin2d_kwargs))
            .mark_rect()
            .encode(x="bin_x_start", y="bin_y_start", color="count")
        )
    elif kind == "hex":
        center = Chart(data).mark_hex(**jk).encode(**enc_center)
    elif kind == "reg":
        # Layered: scatter + smoothed regression line.
        scatter = Chart(data).mark_point(**jk).encode(**enc_center)
        fit = Chart(data).mark_smooth(method="lm", ci=None).encode(x=x, y=y)
        from ferrum.figure.regression import _merge_layers
        center = _merge_layers(scatter, fit)

    # Build top marginal (over x).
    mk = dict(marginal_kws or {})
    enc_top: dict = {"x": x}
    if hue is not None:
        enc_top["color"] = hue
    if marginal_kind == "hist":
        top = Chart(data).mark_histogram(**mk).encode(**enc_top)
    elif marginal_kind == "kde":
        top = Chart(data).mark_density(**mk).encode(**enc_top)
    elif marginal_kind == "rug":
        top = Chart(data).mark_tick(**mk).encode(**enc_top)
    elif marginal_kind == "box":
        top = Chart(data).mark_boxplot(**mk).encode(**enc_top)

    # Build right marginal (over y, oriented vertically).
    enc_right: dict = {"x": y}  # marginal sample is the y-field, but plotted
                                 # on its own coordinate system in the right cell
    if hue is not None:
        enc_right["color"] = hue
    if marginal_kind == "hist":
        right = Chart(data).mark_histogram(**mk).encode(**enc_right)
    elif marginal_kind == "kde":
        right = Chart(data).mark_density(**mk).encode(**enc_right)
    elif marginal_kind == "rug":
        right = Chart(data).mark_tick(**mk).encode(**enc_right)
    elif marginal_kind == "box":
        right = Chart(data).mark_boxplot(**mk).encode(**enc_right)

    if height is not None:
        center = center.properties(width=height, height=height)

    if theme is not None:
        center = center.theme(theme)
        top = top.theme(theme)
        right = right.theme(theme)

    return JointChart(
        center,
        top=top,
        right=right,
        ratio=ratio,
        spacing=space,
    )
