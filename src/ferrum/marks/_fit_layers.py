"""The ribbon + line (+ metrics) layer triple of the fit-family transforms.

`Smooth`, `Robust`, `Logistic` and `Glm` all publish one fitted curve with an
optional confidence band under the same schema -- ``x, y, ci_lower, ci_upper``
(``FIT_OUTPUT_COLUMNS`` in Rust, spec §4.7 / M6c). Two Python paths draw it:
``marks/statistical.desugar_smooth`` for ``mark_smooth``, and
``plots/regression._fit_chart_from_named_transform`` for the ``lmplot``
methods that construct their fit transform directly. They built the same three
layers, over the same four registry columns, with the same magic constants --
the band's ``opacity: 0.3`` and the metrics text's ``align``/``dx``/``dy`` --
which meant the two agreed only by coincidence of copied literals.

This module owns the shape; both callers ask it for the layers and keep their
own job (transform construction, chart assembly). ``tests/
test_fit_layer_parity.py`` pins the two paths against each other.
"""

from __future__ import annotations

from ferrum._layer import _Layer
from ferrum._registry import registry_output_columns
from ferrum.encoding import X, Y

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
    x_title: str,
    y_title: str,
    line_kwargs: dict | None = None,
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
    x_title, y_title : str
        The caller-facing column names the axes are titled with. The layers
        below are bound to the TRANSFORM's output columns (``x``, ``y``,
        ``ci_lower``), and the renderer resolves each axis title from layer 0
        only -- spec-level channel ``title``, then layer-0 channel ``title``,
        then layer-0 field name (``build_axis_input``,
        ``crates/ferrum-core/src/render/prepare/mod.rs``). So a chart whose
        only layers are these titled its value axis ``ci_lower`` -- the
        band's lower bound -- whenever no sibling scatter happened to be
        layer 0 (#199). Both the band and the line carry the titles, because
        which of the two lands at index 0 depends on ``ci``. The metrics
        layer does not: ``line`` is unconditional, so metrics is never first.
        ``y2`` does not either -- the renderer reads only the primary
        positional channel for an axis title.

        Required, with no ``None`` default: both call sites always have the
        caller's field names (``mark_smooth``'s desugar refuses an unencoded
        x or y before reaching here, and ``lmplot``/``regplot`` type ``x``
        and ``y`` as required ``str``), so an optional spelling would only
        add an untestable arm that silently reintroduces the defect.
    line_kwargs : dict, optional
        Mark kwargs the caller aimed at the fit line (``lmplot``'s
        ``line_kws``). Recorded on the line layer's ``caller_style_keys`` as
        well as its ``mark_kwargs``, so the chart-level catch-all
        ``**mark_kwargs`` cannot override them.
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
        layers.append(
            _Layer(
                name="ribbon",
                mark="ribbon",
                encoding={
                    "x": X(cols["x"], title=x_title),
                    "y": Y(cols["ci_lower"], title=y_title),
                    "y2": cols["ci_upper"],
                },
                mark_kwargs=ribbon_kwargs,
                data_source=data_source,
            )
        )

    resolved_line_kwargs = dict(line_kwargs) if line_kwargs else {}
    # The caller's keys, recorded so they beat the chart-level catch-all
    # `**mark_kwargs` (see `_Layer.caller_style_keys`).
    caller_named = frozenset(line_kwargs) if line_kwargs else frozenset()
    layers.append(
        _Layer(
            name="line",
            mark="line",
            encoding={
                "x": X(cols["x"], title=x_title),
                "y": Y(cols["y"], title=y_title),
            },
            mark_kwargs=resolved_line_kwargs or None,
            caller_style_keys=caller_named,
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
