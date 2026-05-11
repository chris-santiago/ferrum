"""Heavy-stat-mark desugar helpers (Phase 8b Sub-batch F).

Each desugar_<name> returns the unified 4-tuple
    (mark, transforms, encoding_remap, synthetic_data)
or 5-tuple for layered:
    ("__layered__", transforms, None, None, layers)
"""
from __future__ import annotations
from typing import Any, Optional

from ferrum import (
    BoxStats, Contour, Hex, Kde2D, QQ, Raster, Swarm, Violin,
)


def desugar_contour(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    thresholds: int = 6,
    smooth: bool = True,
    fill: bool = False,
    cmap: str = "viridis",
    **mark_kwargs: Any,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_contour() requires .encode(x=..., y=...)")
    # Kde2D is UNNAMED so it advances the chain (current → Kde2D output);
    # Contour then runs on the chained Kde2D output. Contour is named so the
    # downstream polygon layer can route through data_source="contour".
    transforms = [
        Kde2D(x=x_field, y=y_field, bandwidth=bandwidth, n=128),
        Contour(thresholds=thresholds, fill=fill, smooth=smooth, name="contour"),
    ]
    layers = [{
        "mark": "polygon",
        "encoding": {"x": "contour_x", "y": "contour_y"},
        "mark_kwargs": {"cmap": cmap, "detail": "level_id"},
        "data_source": "contour",
    }]
    return ("__layered__", transforms, None, None, layers)


def desugar_violin(
    x_field: str | None,
    y_field: str | None,
    *,
    bandwidth: str | float = "scott",
    inner: Optional[str] = "box",
    **mark_kwargs: Any,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_violin() requires .encode(x=..., y=...)")
    if inner not in ("box", "quartile", "point", None):
        raise ValueError(
            f"mark_violin inner must be one of 'box', 'quartile', 'point', or None; got {inner!r}"
        )

    transforms = [Violin(field=y_field, groupby=[x_field], bandwidth=bandwidth, name="violin")]
    violin_layer = {
        "mark": "polygon",
        "encoding": {"x": x_field, "y": "violin_y"},
        "mark_kwargs": {"detail": "group_id", "fill_opacity": 0.5},
        "data_source": "violin",
    }
    if inner is None:
        return ("__layered__", transforms, None, None, [violin_layer])
    if inner == "point":
        return ("__layered__", transforms, None, None,
                [violin_layer, {"mark": "point", "encoding": {"x": x_field, "y": y_field}}])
    if inner == "quartile":
        transforms.append(BoxStats(field=y_field, groupby=[x_field], name="quart"))
        layers = [violin_layer]
        for col in ("q1", "median", "q3"):
            mk = {} if col == "median" else {"stroke_dash": [2, 2]}
            layers.append({"mark": "rule", "encoding": {"x": x_field, "y": col},
                           "mark_kwargs": mk, "data_source": "quart"})
        return ("__layered__", transforms, None, None, layers)
    # inner == "box"
    from ferrum.marks.composite import desugar_boxplot
    _, box_t, _, _, box_layers = desugar_boxplot(x_field, y_field, extent=1.5, outliers=False, size=0.1)
    return ("__layered__", [*transforms, *box_t], None, None, [violin_layer, *box_layers])


def desugar_qq(
    field: str,
    *,
    distribution: str = "normal",
    dequantize: bool = False,
    line: bool = True,
    **mark_kwargs: Any,
) -> tuple:
    if distribution not in ("normal", "uniform", "exponential"):
        raise ValueError(
            f"mark_qq distribution must be 'normal', 'uniform', or 'exponential'; got {distribution!r}"
        )
    transforms = [QQ(field=field, distribution=distribution, dequantize=dequantize,
                     emit_line=line, name="qq_main")]
    layers = [{"mark": "point", "encoding": {"x": "theoretical", "y": "sample"},
               "data_source": "qq_main"}]
    if line:
        layers.append({"mark": "rule",
                       "encoding": {"x": "qq_line_x_start", "y": "qq_line_y_start",
                                    "x2": "qq_line_x_end", "y2": "qq_line_y_end"},
                       "data_source": "qq_line"})
    return ("__layered__", transforms, None, None, layers)


def desugar_raster(
    x_field: str | None,
    y_field: str | None,
    *,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str = "viridis",
    resolution: Any = "screen",
    blend: str = "alpha",
    min_count: Optional[int] = None,
    log_scale: bool = False,
    **mark_kwargs: Any,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_raster() requires .encode(x=..., y=...)")
    if aggregate in ("mean", "sum") and field is None:
        raise ValueError(f"mark_raster aggregate={aggregate!r} requires field=...")
    if blend == "additive":
        from ferrum._warn import warn_once
        warn_once("mark_raster", "blend_additive",
                  "mark_raster blend='additive' deferred to Phase 11; using alpha blending")

    transforms = [Raster(x=x_field, y=y_field, aggregate=aggregate, field=field,
                         resolution=resolution, min_count=min_count, log_scale=log_scale,
                         name="raster")]
    layers = [{
        "mark": "image",
        "encoding": {"x": x_field, "y": y_field},
        "mark_kwargs": {"cmap": cmap},
        "data_source": "raster",
    }]
    return ("__layered__", transforms, None, None, layers)


def desugar_hex(
    x_field: str | None,
    y_field: str | None,
    *,
    bin_size: Optional[float] = None,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str = "viridis",
    stroke: Optional[str] = None,
    stroke_width: float = 0,
    **mark_kwargs: Any,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_hex() requires .encode(x=..., y=...)")
    if aggregate in ("mean", "sum") and field is None:
        raise ValueError(f"mark_hex aggregate={aggregate!r} requires field=...")
    if aggregate not in ("count", "mean", "sum"):
        from ferrum._warn import warn_once
        warn_once("mark_hex", "aggregate_unsupported",
                  f"mark_hex aggregate={aggregate!r} deferred; falling back to 'count'")
        aggregate = "count"
    transforms = [Hex(x=x_field, y=y_field, bin_size=bin_size, aggregate=aggregate, field=field,
                      name="hex")]
    layers = [{
        "mark": "polygon",
        "encoding": {"x": "hex_x", "y": "hex_y"},
        "mark_kwargs": {"cmap": cmap, "detail": "hex_id"},
        "data_source": "hex",
    }]
    return ("__layered__", transforms, None, None, layers)


def desugar_swarm(
    x_field: str | None,
    y_field: str | None,
    *,
    size: int = 4,
    orient: str = "vertical",
    spacing: float = 1.0,
    side: str = "both",
    dodge: Optional[str] = None,
    **mark_kwargs: Any,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_swarm() requires .encode(x=..., y=...)")
    if dodge is not None:
        from ferrum._warn import warn_once
        warn_once("mark_swarm", "dodge",
                  "mark_swarm dodge= is not yet supported; rendering single-group swarm")
    cat = x_field if orient == "vertical" else y_field
    val = y_field if orient == "vertical" else x_field
    transforms = [Swarm(category=cat, value=val, point_size=float(size), spacing=spacing, side=side,
                        name="swarm")]
    if orient == "vertical":
        # Encode the chart's original category & value fields so the ordinal x
        # axis renders properly with the category labels. The Swarm transform
        # also emits a `__pos_x_offset__` column (pixel offset on the cross axis)
        # which the renderer's standard position-offset path applies on top of
        # the category band center — same mechanism Dodge uses (Phase 9c).
        layers = [{
            "mark": "point",
            "encoding": {"x": cat, "y": val},
            "data_source": "swarm",
        }]
    else:
        # TODO(phase-10g+): horizontal-orient swarm still uses the legacy
        # value-axis-data-unit encoding. Fixing it requires either threading
        # orient through the Rust transform so it emits __pos_y_offset__ instead,
        # or adding a column-rename step in the Python pipeline. Lightly tested
        # path; smoke render still produces an SVG.
        layers = [{
            "mark": "point",
            "encoding": {"x": "swarm_x", "y": "swarm_y"},
            "data_source": "swarm",
        }]
    return ("__layered__", transforms, None, None, layers)


def desugar_function(
    fn,
    parent_chart_x_data=None,
    *,
    domain: Optional[tuple] = None,
    n: int = 200,
    clip: bool = True,
    **mark_kwargs: Any,
) -> tuple:
    """The only desugar that materializes a synthetic Arrow table."""
    import numpy as np
    import pyarrow as pa

    if domain is not None:
        d = domain
    elif parent_chart_x_data is not None and len(parent_chart_x_data) > 0:
        d = (float(np.nanmin(parent_chart_x_data)), float(np.nanmax(parent_chart_x_data)))
    else:
        raise ValueError("mark_function requires explicit domain when chart has no other data layers")

    xs = np.linspace(d[0], d[1], n)
    ys = fn(xs)
    if not isinstance(ys, np.ndarray) or ys.shape != (n,):
        raise ValueError(
            f"mark_function callable must return numpy array of shape ({n},); got shape {getattr(ys, 'shape', None)}"
        )

    synthetic = pa.Table.from_pydict({"x": xs, "y": ys})
    return ("line", [], {"x": "x", "y": "y"}, synthetic)
