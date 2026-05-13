"""Joint distribution convenience functions (jointplot)."""

from __future__ import annotations
from typing import Any

from ferrum import Bin2D, Chart, JointChart


_VALID_CENTER_KINDS = {"scatter", "kde", "hist", "hex", "reg"}
_VALID_MARGINAL_KINDS = {"hist", "kde", "rug", "box"}


def jointplot(
    data: Any,
    *,
    x: str,
    y: str,
    hue: Any = None,
    kind: str = "scatter",
    marginal_kind: str = "hist",
    ratio: int = 5,
    space: float = 0.05,
    xlim: Any = None,
    ylim: Any = None,
    joint_kws: Any = None,
    marginal_kws: Any = None,
    height: float | None = None,
    theme: Any = None,
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
        ``(min, max)`` domain override for the x-axis.  Applied as an
        explicit scale domain on the center and top-marginal x encodings
        via ``X(field, scale={"domain": [min, max]})``.
    ylim : tuple, optional
        ``(min, max)`` domain override for the y-axis.  Applied as an
        explicit scale domain on the center and right-marginal y encodings
        via ``Y(field, scale={"domain": [min, max]})``.
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
    from ferrum.encoding import X as _X, Y as _Y

    jk = dict(joint_kws or {})

    enc_center: dict = {"x": x, "y": y}
    if hue is not None:
        enc_center["color"] = hue
    enc_center.update(encode_kwargs)

    # Apply xlim/ylim as scale domain overrides on center encodings.
    if xlim is not None:
        x_field = enc_center.get("x", x)
        if isinstance(x_field, str):
            enc_center["x"] = _X(x_field, scale={"domain": list(xlim)})
    if ylim is not None:
        y_field = enc_center.get("y", y)
        if isinstance(y_field, str):
            enc_center["y"] = _Y(y_field, scale={"domain": list(ylim)})

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
        hist_enc: dict = {"x": "bin_x_start", "y": "bin_y_start", "color": "count"}
        if xlim is not None:
            hist_enc["x"] = _X("bin_x_start", scale={"domain": list(xlim)})
        if ylim is not None:
            hist_enc["y"] = _Y("bin_y_start", scale={"domain": list(ylim)})
        center = (
            Chart(data)
            .transform(Bin2D(x=x, y=y, **bin2d_kwargs))
            .mark_rect()
            .encode(**hist_enc)
        )
    elif kind == "hex":
        center = Chart(data).mark_hex(**jk).encode(**enc_center)
    elif kind == "reg":
        # Layered: scatter + smoothed regression line.
        scatter = Chart(data).mark_point(**jk).encode(**enc_center)
        reg_enc: dict = {"x": x, "y": y}
        if xlim is not None:
            reg_enc["x"] = _X(x, scale={"domain": list(xlim)})
        if ylim is not None:
            reg_enc["y"] = _Y(y, scale={"domain": list(ylim)})
        fit = Chart(data).mark_smooth(method="lm", ci=None).encode(**reg_enc)
        from ferrum.figure.regression import _merge_layers

        center = _merge_layers(scatter, fit)

    # Build top marginal (over x).
    mk = dict(marginal_kws or {})
    enc_top: dict = {"x": x}
    if xlim is not None:
        enc_top["x"] = _X(x, scale={"domain": list(xlim)})
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

    # Build right marginal — oriented horizontally so bars/density grow along
    # the marginal's x-axis while the binned data dimension stays on the
    # marginal's y-axis (shared with the centre cell via share_y).
    enc_right: dict = {"y": y}
    if ylim is not None:
        enc_right["y"] = _Y(y, scale={"domain": list(ylim)})
    if hue is not None:
        enc_right["color"] = hue
    if marginal_kind == "hist":
        right = Chart(data).mark_histogram(orientation="horizontal", **mk).encode(**enc_right)
    elif marginal_kind == "kde":
        right = Chart(data).mark_density(orientation="horizontal", **mk).encode(**enc_right)
    elif marginal_kind == "rug":
        # Tick mark has no bin/density direction — the y-binding is enough.
        right = Chart(data).mark_tick(**mk).encode(**enc_right)
    elif marginal_kind == "box":
        # Boxplot is intrinsically asymmetric across the categorical axis;
        # JointChart uses the default vertical orientation with the data on x
        # (composite-mark orientation work tracked separately).
        right = Chart(data).mark_boxplot(**mk).encode(x=y)

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
