"""The ribbon + line (+ metrics) layer triple of the fit-family transforms.

`Smooth`, `Robust`, `Logistic` and `Glm` all publish one fitted curve with an
optional confidence band under the same schema -- ``x, y, ci_lower, ci_upper``
(``FIT_OUTPUT_COLUMNS`` in Rust, spec §4.7 / M6c). Two Python paths draw it:
``marks/statistical.desugar_smooth`` for ``mark_smooth``, and
``plots/regression._fit_chart_from_named_transform`` for the ``lmplot``
methods that construct their fit transform directly. They built the same three
layers, over the same four registry columns, with the same magic constants --
the band's ``opacity: 0.3`` and the metrics text's ``align``/``dx``/``dy`` --
which meant the two agreed only by coincidence of copied literals, and
``neutral_stroke`` had already landed in one of them alone.

This module owns the shape; both callers ask it for the layers and keep their
own job (transform construction, chart assembly). ``tests/
test_fit_layer_parity.py`` pins the two paths against each other.
"""

from __future__ import annotations

from ferrum._layer import _Layer
from ferrum._registry import registry_output_columns

#: The theme token that keeps a POOLED fit from impersonating a color level.
#: ``marks/composite.py``'s boxplot whisker/median layers use the same one;
#: ``resolve_paint_color`` (``crates/ferrum-core/src/render/draw.rs``) resolves
#: it to the theme's label color, never a scale/palette lookup, so it cannot
#: collide with a color level the way the theme's default accent can.
_NEUTRAL_PAINT = "theme:label"

#: Columns the metrics layer reads. Not a registry lookup: they come from the
#: transform's conditional ``inject_metrics=True`` output, which the registry
#: declares neither as a primary output nor as a named secondary batch (it
#: exists only when that flag is set), so there is no schema to check against.
_METRICS_Y = "_metrics_y"
_METRICS_TEXT = "_metrics_text"


def build_fit_layers(
    kind: str,
    *,
    data_source: str,
    ci: float | None,
    line_kwargs: dict | None = None,
    neutral_stroke: bool = False,
    metrics: bool = False,
) -> list[_Layer]:
    """Build the fit-family layer triple for one fitted-curve transform.

    Parameters
    ----------
    kind : str
        The transform's registry kind (``"smooth"``, ``"robust"``,
        ``"logistic"``, ``"glm"``). Its declared ``outputs`` are what the
        layers' channels are bound to, so a Rust-side rename raises here
        rather than drawing nothing.
    data_source : str
        The named output the layers read -- the transform's own ``name=``.
        Passed separately from *kind* because they are different things that
        happen to coincide at both call sites today.
    ci : float or None
        The confidence level the transform was built with. ``None`` omits the
        band layer; the value itself is not read here (the transform computes
        the interval), only its presence.
    line_kwargs : dict, optional
        Mark kwargs for the fit line (``lmplot``'s ``line_kws``).
    neutral_stroke : bool, default False
        Paint the band and the line with the neutral theme token instead of
        the chart's color resolution. Set when the transform produces ONE
        POOLED fit while a color encoding is active on a sibling layer, so
        the pooled curve cannot visually impersonate one color level. A
        caller-supplied ``line_kwargs["stroke"]`` still wins.
    metrics : bool, default False
        Append the corner R²/RMSE/MAE text layer. Only meaningful when the
        transform was itself built with ``inject_metrics=True``; the columns
        it reads exist only then.

    Returns
    -------
    list of _Layer
        ``[ribbon?, line, metrics?]`` -- the band first so the line draws
        over it.
    """
    cols = registry_output_columns(kind, ["x", "y", "ci_lower", "ci_upper"])
    layers: list[_Layer] = []

    if ci is not None:
        ribbon_kwargs: dict = {"opacity": 0.3}
        if neutral_stroke:
            ribbon_kwargs["fill"] = _NEUTRAL_PAINT
        layers.append(
            _Layer(
                name="ribbon",
                mark="ribbon",
                encoding={"x": cols["x"], "y": cols["ci_lower"], "y2": cols["ci_upper"]},
                mark_kwargs=ribbon_kwargs,
                data_source=data_source,
            )
        )

    resolved_line_kwargs = dict(line_kwargs) if line_kwargs else {}
    if neutral_stroke:
        resolved_line_kwargs.setdefault("stroke", _NEUTRAL_PAINT)
    layers.append(
        _Layer(
            name="line",
            mark="line",
            encoding={"x": cols["x"], "y": cols["y"]},
            mark_kwargs=resolved_line_kwargs or None,
            data_source=data_source,
        )
    )

    if metrics:
        layers.append(
            _Layer(
                name="metrics",
                mark="text",
                encoding={"x": cols["x"], "y": _METRICS_Y, "text": _METRICS_TEXT},
                mark_kwargs={"align": "right", "dx": -4, "dy": 4},
                data_source=data_source,
            )
        )
    return layers
