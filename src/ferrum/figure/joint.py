"""Phase 9e — jointplot."""
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
    """Joint-distribution figure-level function — see ferrum-spec.md §3.14.

    Returns a JointChart with a center scatter/kde/hist/hex/reg chart and
    matching marginal histograms / KDEs / rugs / boxplots along the x and y
    axes.
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
