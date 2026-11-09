"""A statistic refuses a ``bin=``/``aggregate=`` it cannot honor, in the caller's voice.

``mark_histogram``, ``mark_density``, ``mark_smooth`` (and
``mark_point().mark_smooth()``), ``mark_function`` with no parent data
layer, ``displot``'s hist/kde/ecdf kinds, ``jointplot``'s and
``pairplot``'s own marks, ``residplot`` and ``regplot``/``lmplot`` each
build a statistic over x/y columns it reads raw, unfolded, or computes
itself; a channel ``bin=``/``aggregate=`` the statistic never reads would
otherwise be dropped silently when the channel is re-pointed at the
statistic's output, through a keyword, an ``encode=`` entry or a shorthand
string alike. Every refusal is raised as an
:class:`~ferrum._entry.EntryRefusal`, so it speaks as the public call the
caller made, never the inner mark or its own parameters, and names a
remedy that renders -- in the voicing call's own spelling when that knob
has one (``displot``'s ``bins=``, ``jointplot``'s ``marginal_kws=``), the
knob-free remedy otherwise. The honored rows render byte-identically to a
sibling route without the option, or to a committed digest of the same
chart.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y
from tests._script_loader import load_script
from tests._svg_extents import axis_tick_labels, x_axis_title, y_axis_title
from tests._svg_rects import data_bar_rects, polylines, step_after_paths

_HIST_ROWS = pl.DataFrame({"v": [float(i) for i in range(1, 21)]})
_XY_ROWS = pl.DataFrame(
    {
        "v": [float(i) for i in range(1, 21)],
        "w": [float(i) * 2.0 + 1.0 for i in range(1, 21)],
    }
)
#: A binary response, for the logistic/glm-routed fit cases.
_XB_ROWS = _XY_ROWS.with_columns((pl.col("w") > 20.0).cast(pl.Int64).alias("b"))
#: Grouped rows, for the histogram's normalized-stack cases.
_GROUPED_HIST_ROWS = _HIST_ROWS.with_columns(g=(pl.col("v") > 10.0))
#: A third numeric column, for mark_hex's color-aggregate fold cases.
_XYZ_ROWS = _XY_ROWS.with_columns(z=(pl.col("v") * 3.0 - 1.0))


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


def _equal_width_binned_mean(rows: pl.DataFrame, *, bins: int) -> pl.DataFrame:
    """*rows* (``v``, ``w``) pre-aggregated into *bins* equal-width bins over
    ``[v.min(), v.max()]``, each bin's row the group's mean ``v`` and ``w`` --
    the same binning ``Smooth(x_bins=...)`` pre-aggregates its input with.
    """
    vmin, vmax = rows["v"].min(), rows["v"].max()
    width = (vmax - vmin) / bins
    edges = [vmin + i * width for i in range(1, bins)]
    return (
        rows.with_columns(bin=pl.col("v").cut(edges, labels=[str(i) for i in range(bins)]))
        .group_by("bin", maintain_order=True)
        .agg(v=pl.col("v").mean(), w=pl.col("w").mean())
        .sort("bin")
    )


def _message(voice: str, written: str, kwarg: str, reason: str, remedy: str) -> str:
    return f"{voice}({written}) is not supported: {reason}, so {kwarg}= would be ignored. {remedy}"


# ---------------------------------------------------------------------------
# An expected-message table, independent of the implementation: each helper
# below computes the (written, reason, remedy) triple a route must produce,
# from the route's own facts (field, axis, door), not by calling the
# refusal owner.
# ---------------------------------------------------------------------------


def _quoted(value: Any) -> str:
    """The repr a refusal quotes *value* with: a literal's own repr, a
    pre-built object's class name alone (``Bin(...)``)."""
    if isinstance(value, (bool, str, int, float, dict, list, tuple)):
        return repr(value)
    return f"{type(value).__name__}(...)"


def _written(channel: str, cls: str, field: str | None, kwarg: str, value: Any) -> str:
    quoted = _quoted(value)
    if field:
        return f"{channel}={cls}({field!r}, {kwarg}={quoted})"
    return f"{channel}={cls}({kwarg}={quoted})"


def _generic_bin_remedy(field: str | None, *, call_kwargs: str | None = None) -> str:
    if not field:
        return "Drop bin=."
    with_clause = f" with {call_kwargs}" if call_kwargs else ""
    return (
        f"Bin the field per row with .transform(fm.Bin({field!r}, per_row=True)) "
        f"and bind its 'bin_start' column{with_clause}, or drop bin=."
    )


def _generic_aggregate_remedy(field: str | None, *, call_kwargs: str | None = None) -> str:
    if not field:
        return "Drop aggregate=."
    with_clause = f" with {call_kwargs}" if call_kwargs else ""
    return (
        "Compute the aggregate with .transform(fm.transform_aggregate(...)) and bind its "
        f"output column{with_clause}, or drop aggregate=."
    )


def _hist_value_axis_remedy(
    kwarg: str,
    field: str | None,
    value_axis: str,
    bin_field: str | None,
    bin_axis: str,
    op: str | None,
) -> str:
    if not field:
        return "Drop bin=." if kwarg == "bin" else "Drop aggregate=."
    va, ba = value_axis.upper(), bin_axis.upper()
    drop_clause = f"drop {value_axis}= (style the axis with fm.{va}(title=...) and no field)"
    if not bin_field:
        return f"Bind the field to bin on {bin_axis}= and {drop_clause}."
    if kwarg == "aggregate":
        return (
            f"Draw the aggregate per bin with a bar mark, fm.Chart(data).mark_bar().encode("
            f"{bin_axis}=fm.{ba}({bin_field!r}, bin=True), {value_axis}=fm.{va}({field!r}, "
            f"aggregate={op!r})), or {drop_clause}."
        )
    return (
        f"Count rows over both fields with fm.Chart(data).mark_hex().encode("
        f"{bin_axis}={bin_field!r}, {value_axis}={field!r}), or {drop_clause}."
    )


_CLS = {"x": "X", "y": "Y"}


# ---------------------------------------------------------------------------
# Generated cases: each generator builds the cells its door's own cell set
# names, as a product over the door's route axes, so a cell cannot be left
# out the way a hand-typed list can.
# ---------------------------------------------------------------------------

_Case = tuple[str, Callable[[], Any], str]


def _axis_channel(axis: str, field: str | None, **kwargs: Any) -> X | Y:
    """Build an ``X``/``Y`` channel for *axis* ("x"/"y"), field-bearing or not."""
    cls = {"x": X, "y": Y}[axis]
    return cls(field, **kwargs) if field is not None else cls(**kwargs)


def _histogram_cases(door: str) -> list[_Case]:
    """mark_histogram/displot(kind='hist') over orientation x its full cell set.

    ``encode`` is built once, keyed by the real channel name (``"x"``,
    ``"y"``, ``"color"``), so the same dict drives three doors:
    ``Chart.mark_histogram(...).encode(**encode)``,
    ``fm.displot(rows, kind="hist", **encode, **mark_kwargs)`` and
    ``fm.displot(rows, x="v", kind="hist", **mark_kwargs).encode(**encode)``
    (``door="displot_posthoc"``, voiced as ``displot`` like the keyword
    door, reached once the pending histogram mark resolves at render time
    -- the same backstop the ECDF/kde post-hoc doors exercise). displot has
    no orientation of its own (vertical only) and no normalized-stack
    spelling beyond its own ``multiple='fill'`` (``position=``/channel
    ``stack=`` are ``mark_histogram``-only keywords).
    """
    cases: list[_Case] = []
    voice = "mark_histogram" if door == "mark_histogram" else "displot"
    orientations = ("vertical", "horizontal") if door == "mark_histogram" else ("vertical",)
    bin_remedy = {
        "mark_histogram": "State the rule with mark_histogram's bins=, bin_count= or bin_width=, or drop bin=.",
        "displot": "State the rule with displot's bins=, or drop bin=.",
    }[voice]
    density_spelling, density_remedy = {
        "mark_histogram": (
            "density=True",
            "Drop aggregate=, or draw the count with density=False.",
        ),
        "displot": ("stat='density'", "Drop aggregate=, or draw the count with stat='count'."),
    }[voice]
    share_remedy = "Drop aggregate=, or draw the count with multiple='stack'."

    for orientation in orientations:
        bin_axis, value_axis = ("y", "x") if orientation == "horizontal" else ("x", "y")
        bin_cls, value_cls = _CLS[bin_axis], _CLS[value_axis]
        suffix = f"_{orientation}" if door == "mark_histogram" else ""

        def build(
            mark_kwargs: dict,
            encode: dict,
            *,
            grouped: bool = False,
            needs_w: bool = False,
            orientation: str = orientation,
        ):
            def run() -> None:
                if grouped:
                    rows = _GROUPED_HIST_ROWS
                elif needs_w:
                    rows = _XY_ROWS
                else:
                    rows = _HIST_ROWS
                if door == "mark_histogram":
                    fm.Chart(rows).mark_histogram(orientation=orientation, **mark_kwargs).encode(
                        **encode
                    ).to_svg()
                elif door == "displot":
                    fm.displot(rows, kind="hist", **encode, **mark_kwargs)
                else:
                    fm.displot(rows, x="v", kind="hist", **mark_kwargs).encode(**encode).to_svg()

            return run

        def case(
            cell_id: str,
            mark_kwargs: dict,
            encode: dict,
            refused_axis: str,
            kwarg: str,
            value: Any,
            reason: str,
            remedy: str,
            *,
            grouped: bool = False,
            needs_w: bool = False,
        ):
            refused_cls = _CLS[refused_axis]
            field = encode[refused_axis].field if hasattr(encode[refused_axis], "field") else None
            cases.append(
                (
                    f"{door}_{cell_id}{suffix}",
                    build(mark_kwargs, encode, grouped=grouped, needs_w=needs_w),
                    _message(
                        voice,
                        _written(refused_axis, refused_cls, field, kwarg, value),
                        kwarg,
                        reason,
                        remedy,
                    ),
                )
            )

        # Binned axis: a rule of its own, as a dict and as a pre-built fm.Bin.
        for label, bin_value in (
            ("bin_rule_dict", {"bin_count": 5}),
            ("bin_rule_binobj", fm.Bin("v", bin_count=5)),
        ):
            case(
                label,
                {},
                {bin_axis: _axis_channel(bin_axis, "v", bin=bin_value)},
                bin_axis,
                "bin",
                bin_value,
                f"the histogram bins {bin_axis} by its own rule",
                bin_remedy,
            )

        # Binned axis: aggregate=.
        case(
            "binned_axis_aggregate",
            {},
            {bin_axis: _axis_channel(bin_axis, "v", aggregate="mean")},
            bin_axis,
            "aggregate",
            "mean",
            f"the histogram bins {bin_axis} rather than aggregating it",
            _generic_aggregate_remedy("v"),
        )

        # Value axis: field-bearing bin=True, aggregate='mean', aggregate='count'.
        for label, kwarg, value in (
            ("value_axis_field_bin", "bin", True),
            ("value_axis_field_mean", "aggregate", "mean"),
            ("value_axis_field_count", "aggregate", "count"),
        ):
            op = value if kwarg == "aggregate" else None
            case(
                label,
                {},
                {bin_axis: "v", value_axis: _axis_channel(value_axis, "w", **{kwarg: value})},
                value_axis,
                kwarg,
                value,
                f"the histogram computes {value_axis} itself",
                _hist_value_axis_remedy(kwarg, "w", value_axis, "v", bin_axis, op),
                needs_w=True,
            )

        # Value axis: field-bearing bin=True / aggregate='mean' with no field
        # bound on the binned axis (mark_histogram only: displot refuses a
        # call without x= before it reads either channel).
        if door == "mark_histogram":
            for label, kwarg, value in (
                ("value_axis_field_bin_no_binned_field", "bin", True),
                ("value_axis_field_mean_no_binned_field", "aggregate", "mean"),
            ):
                op = value if kwarg == "aggregate" else None
                case(
                    label,
                    {},
                    {value_axis: _axis_channel(value_axis, "w", **{kwarg: value})},
                    value_axis,
                    kwarg,
                    value,
                    f"the histogram computes {value_axis} itself",
                    _hist_value_axis_remedy(kwarg, "w", value_axis, None, bin_axis, op),
                    needs_w=True,
                )

        # Value axis: fieldless bin=True, fieldless aggregate='mean' (not
        # 'count', so always refused regardless of density/share).
        for label, kwarg, value, remedy in (
            ("value_axis_fieldless_bin", "bin", True, "Drop bin=."),
            ("value_axis_fieldless_mean", "aggregate", "mean", "Drop aggregate=."),
        ):
            case(
                label,
                {},
                {bin_axis: "v", value_axis: _axis_channel(value_axis, None, **{kwarg: value})},
                value_axis,
                kwarg,
                value,
                f"the histogram computes {value_axis} itself",
                remedy,
            )

        # Fieldless count under density.
        density_kwarg = {"density": True} if door == "mark_histogram" else {"stat": "density"}
        case(
            "count_under_density",
            density_kwarg,
            {bin_axis: "v", value_axis: _axis_channel(value_axis, None, aggregate="count")},
            value_axis,
            "aggregate",
            "count",
            f"with {density_spelling} the histogram draws density on {value_axis}, not a count",
            density_remedy,
        )

        # Fieldless count under multiple='fill' (both doors).
        case(
            "count_under_fill",
            {"multiple": "fill"}
            if door != "mark_histogram"
            else {"multiple": "fill", "groupby": ["g"]},
            {
                bin_axis: "v",
                "color": "g",
                value_axis: _axis_channel(value_axis, None, aggregate="count"),
            }
            if door == "mark_histogram"
            else {bin_axis: "v", value_axis: _axis_channel(value_axis, None, aggregate="count")},
            value_axis,
            "aggregate",
            "count",
            f"with multiple='fill' the histogram draws each bin's share of its stack on {value_axis}, not a count",
            share_remedy,
            grouped=(door == "mark_histogram"),
        )

        if door == "mark_histogram":
            # Fieldless count under a caller's own position=fm.Stack(offset='normalize').
            case(
                "count_under_position_normalize",
                {"groupby": ["g"], "position": fm.Stack(offset="normalize")},
                {
                    bin_axis: "v",
                    "color": "g",
                    value_axis: _axis_channel(value_axis, None, aggregate="count"),
                },
                value_axis,
                "aggregate",
                "count",
                (
                    "with position=fm.Stack(offset='normalize') the histogram draws each bin's "
                    f"share of its stack on {value_axis}, not a count"
                ),
                "Drop aggregate=, or draw the count with position=fm.Stack().",
                grouped=True,
            )

            # Fieldless count under the value channel's own stack='normalize'.
            case(
                "count_under_stack_normalize",
                {"groupby": ["g"]},
                {
                    bin_axis: "v",
                    "color": "g",
                    value_axis: _axis_channel(
                        value_axis, None, aggregate="count", stack="normalize"
                    ),
                },
                value_axis,
                "aggregate",
                "count",
                f"with stack='normalize' the histogram draws each bin's share of its stack on {value_axis}, not a count",
                "Drop aggregate=, or draw the count with stack=True.",
                grouped=True,
            )

    return cases


def _smooth_cases() -> list[_Case]:
    """mark_smooth over its three entry shapes x its full x/y x bin/aggregate cell set."""
    cases: list[_Case] = []

    def chart_for(form: str):
        if form == "single_line":
            return lambda: fm.Chart(_XY_ROWS).mark_smooth()
        if form == "ci":
            return lambda: fm.Chart(_XY_ROWS).mark_smooth(ci=0.95)
        return lambda: fm.Chart(_XY_ROWS).mark_point().encode(x="v", y="w").mark_smooth()

    for form in ("single_line", "ci", "chained"):
        for channel, field, reason in (
            ("x", "v", "the fit reads the raw predictor"),
            ("y", "w", "the fit reads the raw response"),
        ):
            for kwarg, value, remedy in (
                (
                    "bin",
                    True,
                    "Bin the predictor with x_bins=<number of bins>, the fit's own binning, or drop bin=."
                    if channel == "x"
                    else _generic_bin_remedy(field),
                ),
                ("aggregate", "mean", _generic_aggregate_remedy(field)),
            ):
                other_channel = "y" if channel == "x" else "x"
                other_field = "w" if channel == "x" else "v"

                def build(
                    form=form,
                    channel=channel,
                    field=field,
                    other_channel=other_channel,
                    other_field=other_field,
                    kwarg=kwarg,
                    value=value,
                ):
                    def run() -> None:
                        chart = chart_for(form)()
                        ch = (
                            X(field, **{kwarg: value})
                            if channel == "x"
                            else Y(field, **{kwarg: value})
                        )
                        chart.encode(**{channel: ch, other_channel: other_field}).to_svg()

                    return run

                cases.append(
                    (
                        f"mark_smooth_{form}_{channel}_{kwarg}",
                        build(),
                        _message(
                            "mark_smooth",
                            _written(channel, _CLS[channel], field, kwarg, value),
                            kwarg,
                            reason,
                            remedy,
                        ),
                    )
                )
    return cases


def _fit_cases() -> list[_Case]:
    """lmplot/regplot over method x x/y x bin/aggregate."""
    cases: list[_Case] = []
    methods_data = {
        "lm": _XY_ROWS,
        "logistic": _XB_ROWS,
        "glm": _XB_ROWS,
        "robust": _XY_ROWS,
    }
    y_field = {"lm": "w", "logistic": "b", "glm": "b", "robust": "w"}
    smooth_routed = {"lm"}

    for door in ("lmplot", "regplot"):
        call = fm.lmplot if door == "lmplot" else fm.regplot
        for method, rows in methods_data.items():
            yf = y_field[method]
            for channel, field, reason in (
                ("x", "v", "the fit reads the raw predictor"),
                ("y", yf, "the fit reads the raw response"),
            ):
                for kwarg, value in (("bin", True), ("aggregate", "mean")):
                    if channel == "x" and kwarg == "bin":
                        remedy = (
                            "Bin the predictor with x_bins=<number of bins>, the fit's own binning, or drop bin=."
                            if method in smooth_routed
                            else _generic_bin_remedy(field)
                        )
                    elif kwarg == "bin":
                        remedy = _generic_bin_remedy(field)
                    else:
                        remedy = _generic_aggregate_remedy(field)
                    other_channel = "y" if channel == "x" else "x"
                    other_field = yf if channel == "x" else "v"

                    def build(
                        call=call,
                        rows=rows,
                        method=method,
                        channel=channel,
                        field=field,
                        kwarg=kwarg,
                        value=value,
                        other_channel=other_channel,
                        other_field=other_field,
                    ):
                        def run() -> None:
                            ch = (
                                X(field, **{kwarg: value})
                                if channel == "x"
                                else Y(field, **{kwarg: value})
                            )
                            call(rows, method=method, **{channel: ch, other_channel: other_field})

                        return run

                    cases.append(
                        (
                            f"{door}_{method}_{channel}_{kwarg}",
                            build(),
                            _message(
                                door,
                                _written(channel, _CLS[channel], field, kwarg, value),
                                kwarg,
                                reason,
                                remedy,
                            ),
                        )
                    )

    # Post-hoc twins (scatter=False, method='lm'): the same remedies as the
    # keyword lm cells, reached through a `.encode()` written after the call.
    for door in ("lmplot", "regplot"):
        call = fm.lmplot if door == "lmplot" else fm.regplot
        for channel, field, reason in (
            ("x", "v", "the fit reads the raw predictor"),
            ("y", "w", "the fit reads the raw response"),
        ):
            for kwarg, value in (("bin", True), ("aggregate", "mean")):
                remedy = (
                    "Bin the predictor with x_bins=<number of bins>, the fit's own binning, or drop bin=."
                    if channel == "x" and kwarg == "bin"
                    else (
                        _generic_bin_remedy(field)
                        if kwarg == "bin"
                        else _generic_aggregate_remedy(field)
                    )
                )

                def build(call=call, channel=channel, field=field, kwarg=kwarg, value=value):
                    def run() -> None:
                        ch = (
                            X(field, **{kwarg: value})
                            if channel == "x"
                            else Y(field, **{kwarg: value})
                        )
                        call(_XY_ROWS, x="v", y="w", method="lm", scatter=False).encode(
                            **{channel: ch}
                        ).to_svg()

                    return run

                cases.append(
                    (
                        f"{door}_posthoc_lm_{channel}_{kwarg}",
                        build(),
                        _message(
                            door,
                            _written(channel, _CLS[channel], field, kwarg, value),
                            kwarg,
                            reason,
                            remedy,
                        ),
                    )
                )
    return cases


def _kde_cases() -> list[_Case]:
    """mark_density (univariate x, bivariate y), displot(kind='kde') keyword and post-hoc,
    over x/y x bin/aggregate.
    """
    cases: list[_Case] = []
    reason = "the kernel density estimate reads raw rows"
    for door in ("mark_density", "displot", "displot_posthoc"):
        voice = "mark_density" if door == "mark_density" else "displot"
        for channel, field, other, other_field in (("x", "v", "y", None), ("y", "w", "x", "v")):
            for kwarg, value in (("bin", True), ("aggregate", "mean")):
                remedy = (
                    _generic_bin_remedy(field)
                    if kwarg == "bin"
                    else _generic_aggregate_remedy(field)
                )

                def build(
                    door=door,
                    channel=channel,
                    field=field,
                    other=other,
                    other_field=other_field,
                    kwarg=kwarg,
                    value=value,
                ):
                    def run() -> None:
                        ch = (
                            X(field, **{kwarg: value})
                            if channel == "x"
                            else Y(field, **{kwarg: value})
                        )
                        encode = {channel: ch}
                        if other_field is not None:
                            encode[other] = other_field
                        if door == "mark_density":
                            fm.Chart(_XY_ROWS).mark_density().encode(**encode).to_svg()
                        elif door == "displot":
                            fm.displot(_XY_ROWS, kind="kde", **encode)
                        else:
                            fm.displot(_XY_ROWS, kind="kde", x="v").encode(**encode).to_svg()

                    return run

                cases.append(
                    (
                        f"{door}_kde_{channel}_{kwarg}",
                        build(),
                        _message(
                            voice,
                            _written(channel, _CLS[channel], field, kwarg, value),
                            kwarg,
                            reason,
                            remedy,
                        ),
                    )
                )
    return cases


def _contour_cases() -> list[_Case]:
    """mark_contour (fill=False and fill=True both) over x/y x bin/aggregate.

    The 2-D kernel density estimate reads raw rows exactly as mark_density's
    bivariate path does, so this shares that reason and both generic
    remedies.
    """
    cases: list[_Case] = []
    reason = "the kernel density estimate reads raw rows"
    for fill in (False, True):
        for channel, field, other, other_field in (("x", "v", "y", "w"), ("y", "w", "x", "v")):
            for kwarg, value in (("bin", True), ("aggregate", "mean")):
                remedy = (
                    _generic_bin_remedy(field)
                    if kwarg == "bin"
                    else _generic_aggregate_remedy(field)
                )

                def build(
                    fill=fill,
                    channel=channel,
                    field=field,
                    other=other,
                    other_field=other_field,
                    kwarg=kwarg,
                    value=value,
                ):
                    def run() -> None:
                        ch = _axis_channel(channel, field, **{kwarg: value})
                        encode = {channel: ch, other: other_field}
                        fm.Chart(_XY_ROWS).mark_contour(fill=fill).encode(**encode).to_svg()

                    return run

                cases.append(
                    (
                        f"mark_contour_fill_{fill}_{channel}_{kwarg}",
                        build(),
                        _message(
                            "mark_contour",
                            _written(channel, _CLS[channel], field, kwarg, value),
                            kwarg,
                            reason,
                            remedy,
                        ),
                    )
                )
    return cases


_HEX_BIN_REMEDY = "Size the cells with mark_hex's bin_size=, or drop bin=."


def _hex_cases() -> list[_Case]:
    """mark_hex over x/y x bin (incl. bin=True, which hex never draws the 1-D
    bins of) and aggregate.
    """
    cases: list[_Case] = []
    reason = "the hexagons bin x and y by their own grid"
    bin_values: list[Any] = [True, {"bin_count": 3}]
    for channel, field, other, other_field in (("x", "v", "y", "w"), ("y", "w", "x", "v")):
        for value in bin_values:

            def build_bin(
                channel=channel, field=field, other=other, other_field=other_field, value=value
            ):
                def run() -> None:
                    ch = _axis_channel(channel, field, bin=value)
                    encode = {channel: ch, other: other_field}
                    fm.Chart(_XY_ROWS).mark_hex().encode(**encode).to_svg()

                return run

            cases.append(
                (
                    f"mark_hex_{channel}_bin_{type(value).__name__}",
                    build_bin(),
                    _message(
                        "mark_hex",
                        _written(channel, _CLS[channel], field, "bin", value),
                        "bin",
                        reason,
                        _HEX_BIN_REMEDY,
                    ),
                )
            )

        def build_aggregate(channel=channel, field=field, other=other, other_field=other_field):
            def run() -> None:
                ch = _axis_channel(channel, field, aggregate="mean")
                encode = {channel: ch, other: other_field}
                fm.Chart(_XY_ROWS).mark_hex().encode(**encode).to_svg()

            return run

        cases.append(
            (
                f"mark_hex_{channel}_aggregate",
                build_aggregate(),
                _message(
                    "mark_hex",
                    _written(channel, _CLS[channel], field, "aggregate", "mean"),
                    "aggregate",
                    reason,
                    _generic_aggregate_remedy(field),
                ),
            )
        )
    return cases


_RASTER_BIN_REMEDY = "Set the grid with mark_raster's resolution=, or drop bin=."


def _raster_cases() -> list[_Case]:
    """mark_raster over x/y x bin/aggregate."""
    cases: list[_Case] = []
    reason = "the raster bins x and y onto its own pixel grid"
    for channel, field, other, other_field in (("x", "v", "y", "w"), ("y", "w", "x", "v")):
        for kwarg, value in (("bin", True), ("aggregate", "mean")):
            remedy = _RASTER_BIN_REMEDY if kwarg == "bin" else _generic_aggregate_remedy(field)

            def build(
                channel=channel,
                field=field,
                other=other,
                other_field=other_field,
                kwarg=kwarg,
                value=value,
            ):
                def run() -> None:
                    ch = _axis_channel(channel, field, **{kwarg: value})
                    encode = {channel: ch, other: other_field}
                    fm.Chart(_XY_ROWS).mark_raster().encode(**encode).to_svg()

                return run

            cases.append(
                (
                    f"mark_raster_{channel}_{kwarg}",
                    build(),
                    _message(
                        "mark_raster",
                        _written(channel, _CLS[channel], field, kwarg, value),
                        kwarg,
                        reason,
                        remedy,
                    ),
                )
            )
    return cases


def _qq_cases() -> list[_Case]:
    """mark_qq over x (the raw sample, field-bearing) and y (its own computed
    sample quantiles, fieldless) x bin/aggregate.
    """
    cases: list[_Case] = []
    x_reason = "the Q-Q plot reads the raw sample"
    for kwarg, value in (("bin", True), ("aggregate", "mean")):
        remedy = _generic_bin_remedy("v") if kwarg == "bin" else _generic_aggregate_remedy("v")

        def build_x(kwarg=kwarg, value=value):
            def run() -> None:
                fm.Chart(_HIST_ROWS).mark_qq().encode(x=X("v", **{kwarg: value})).to_svg()

            return run

        cases.append(
            (
                f"mark_qq_x_{kwarg}",
                build_x(),
                _message("mark_qq", _written("x", "X", "v", kwarg, value), kwarg, x_reason, remedy),
            )
        )

    y_reason = "the Q-Q plot draws its own sample quantiles on y"
    for kwarg, value in (("bin", True), ("aggregate", "mean")):
        remedy = "Drop bin=." if kwarg == "bin" else "Drop aggregate=."

        def build_y(kwarg=kwarg, value=value):
            def run() -> None:
                fm.Chart(_HIST_ROWS).mark_qq().encode(x="v", y=Y(**{kwarg: value})).to_svg()

            return run

        cases.append(
            (
                f"mark_qq_y_{kwarg}",
                build_y(),
                _message(
                    "mark_qq", _written("y", "Y", None, kwarg, value), kwarg, y_reason, remedy
                ),
            )
        )
    return cases


def _ecdf_cases() -> list[_Case]:
    """displot(kind='ecdf') over x (field-bearing) and y (fieldless) x every bin/aggregate spelling."""
    cases: list[_Case] = []
    x_bin_values = [True, {"bin_count": 3}, fm.Bin("v", bin_count=3)]
    y_bin_values = [True, {"bin_count": 3}, fm.Bin("q", bin_count=3)]

    for value in x_bin_values:

        def build(value=value):
            def run() -> None:
                fm.displot(_HIST_ROWS, x=X("v", bin=value), kind="ecdf")

            return run

        cases.append(
            (
                f"ecdf_x_bin_{type(value).__name__}",
                build(),
                _message(
                    "displot",
                    _written("x", "X", "v", "bin", value),
                    "bin",
                    "the ECDF counts the raw values of x",
                    _generic_bin_remedy("v"),
                ),
            )
        )
    cases.append(
        (
            "ecdf_x_aggregate",
            lambda: fm.displot(_HIST_ROWS, x=X("v", aggregate="mean"), kind="ecdf"),
            _message(
                "displot",
                _written("x", "X", "v", "aggregate", "mean"),
                "aggregate",
                "the ECDF counts the raw values of x",
                _generic_aggregate_remedy("v"),
            ),
        )
    )

    for value in y_bin_values:

        def build(value=value):
            def run() -> None:
                fm.displot(_HIST_ROWS, x="v", y=Y(bin=value), kind="ecdf")

            return run

        cases.append(
            (
                f"ecdf_y_bin_{type(value).__name__}",
                build(),
                _message(
                    "displot",
                    _written("y", "Y", None, "bin", value),
                    "bin",
                    "the ECDF draws its cumulative count on y",
                    "Drop bin=.",
                ),
            )
        )
    cases.append(
        (
            "ecdf_y_aggregate",
            lambda: fm.displot(_HIST_ROWS, x="v", y=Y(aggregate="mean"), kind="ecdf"),
            _message(
                "displot",
                _written("y", "Y", None, "aggregate", "mean"),
                "aggregate",
                "the ECDF draws its cumulative count on y",
                "Drop aggregate=.",
            ),
        )
    )

    # Post-hoc rows: the same refusals, reaching through a `.encode()`
    # written after the call, when the pending mark resolves -- the
    # resolve-time backstop (`_prep_refuse_ecdf_statistics`) that covers
    # the door the call-time pre-check cannot reach. x's full bin-value
    # product gets a twin, same as the call-time cells above.
    for value in x_bin_values:

        def build(value=value):
            def run() -> None:
                fm.displot(_HIST_ROWS, x="v", kind="ecdf").encode(x=X("v", bin=value)).to_svg()

            return run

        cases.append(
            (
                f"ecdf_x_bin_{type(value).__name__}_posthoc",
                build(),
                _message(
                    "displot",
                    _written("x", "X", "v", "bin", value),
                    "bin",
                    "the ECDF counts the raw values of x",
                    _generic_bin_remedy("v"),
                ),
            )
        )
    for value in y_bin_values:

        def build(value=value):
            def run() -> None:
                fm.displot(_HIST_ROWS, x="v", kind="ecdf").encode(y=Y(bin=value)).to_svg()

            return run

        cases.append(
            (
                f"ecdf_y_bin_{type(value).__name__}_posthoc",
                build(),
                _message(
                    "displot",
                    _written("y", "Y", None, "bin", value),
                    "bin",
                    "the ECDF draws its cumulative count on y",
                    "Drop bin=.",
                ),
            )
        )
    cases.append(
        (
            "ecdf_x_aggregate_posthoc",
            lambda: (
                fm.displot(_HIST_ROWS, x="v", kind="ecdf")
                .encode(x=X("v", aggregate="mean"))
                .to_svg()
            ),
            _message(
                "displot",
                _written("x", "X", "v", "aggregate", "mean"),
                "aggregate",
                "the ECDF counts the raw values of x",
                _generic_aggregate_remedy("v"),
            ),
        )
    )
    cases.append(
        (
            "ecdf_y_field_posthoc",
            lambda: fm.displot(_XY_ROWS, x="v", kind="ecdf").encode(y="w").to_svg(),
            "displot(kind='ecdf', y='w') is not supported -- kind='ecdf' draws the "
            "cumulative count of the variable x= names on y, so a second variable "
            "has nowhere to go. Drop y=, or style the count axis with "
            "fm.Y(title=..., scale=...) and no field.",
        )
    )
    cases.append(
        (
            "ecdf_y_aggregate_posthoc",
            lambda: (
                fm.displot(_HIST_ROWS, x="v", kind="ecdf").encode(y=Y(aggregate="mean")).to_svg()
            ),
            _message(
                "displot",
                _written("y", "Y", None, "aggregate", "mean"),
                "aggregate",
                "the ECDF draws its cumulative count on y",
                "Drop aggregate=.",
            ),
        )
    )
    return cases


def _residplot_cases() -> list[_Case]:
    cases: list[_Case] = []
    reason = "the residual fit reads the raw predictor and response"
    for channel, field, other, other_field in (("x", "v", "y", "w"), ("y", "w", "x", "v")):
        for kwarg, value in (("bin", True), ("aggregate", "mean")):
            remedy = (
                _generic_bin_remedy(field, call_kwargs="dropna=False")
                if kwarg == "bin"
                else _generic_aggregate_remedy(field, call_kwargs="dropna=False")
            )

            def build(
                channel=channel,
                field=field,
                other=other,
                other_field=other_field,
                kwarg=kwarg,
                value=value,
            ):
                def run() -> None:
                    ch = (
                        X(field, **{kwarg: value}) if channel == "x" else Y(field, **{kwarg: value})
                    )
                    fm.residplot(_XY_ROWS, **{channel: ch, other: other_field})

                return run

            cases.append(
                (
                    f"residplot_{channel}_{kwarg}",
                    build(),
                    _message(
                        "residplot",
                        _written(channel, _CLS[channel], field, kwarg, value),
                        kwarg,
                        reason,
                        remedy,
                    ),
                )
            )
    return cases


def _mark_function_cases() -> list[_Case]:
    cases: list[_Case] = []
    reason = "fn is drawn over its own grid and no data layer reads the field"

    def remedy_for(kwarg: str) -> str:
        return (
            "Draw the data with a mark before mark_function "
            f"(chart.mark_point().encode(...).mark_function(fn)), which honors {kwarg}=, or drop {kwarg}=."
        )

    for channel, field in (("x", "v"), ("y", "v")):
        for kwarg, value in (("bin", True), ("aggregate", "mean")):

            def build(channel=channel, field=field, kwarg=kwarg, value=value):
                def run() -> None:
                    ch = (
                        X(field, **{kwarg: value}) if channel == "x" else Y(field, **{kwarg: value})
                    )
                    fm.Chart(None).encode(**{channel: ch}).mark_function(
                        lambda xs: xs, domain=[0.0, 10.0]
                    ).to_svg()

                return run

            cases.append(
                (
                    f"mark_function_alone_{channel}_{kwarg}",
                    build(),
                    _message(
                        "mark_function",
                        _written(channel, _CLS[channel], field, kwarg, value),
                        kwarg,
                        reason,
                        remedy_for(kwarg),
                    ),
                )
            )
    return cases


def _spelling_cases() -> list[_Case]:
    """Every figure door, over spelling {shorthand, encode=}, generated the same way
    for each door's own voice and reason/remedy.
    """
    cases: list[_Case] = []

    # door_id -> (call, voice, bin_value, bin_reason, bin_remedy, agg_reason, agg_remedy).
    # `bin_value` is a rule (not bare True) for displot_hist, whose own binned
    # axis honors a redundant bin=True; every other door refuses bin=True
    # itself, so True exercises the same cell there.
    fit_bin_remedy = (
        "Bin the predictor with x_bins=<number of bins>, the fit's own binning, or drop bin=."
    )
    doors: dict[str, tuple[Callable[[Any], Any], str, Any, str, str, str, str]] = {
        "displot_hist": (
            lambda **kw: fm.displot(_HIST_ROWS, kind="hist", **kw),
            "displot",
            {"bin_count": 3},
            "the histogram bins x by its own rule",
            "State the rule with displot's bins=, or drop bin=.",
            "the histogram bins x rather than aggregating it",
            _generic_aggregate_remedy("v"),
        ),
        "displot_kde": (
            lambda **kw: fm.displot(_HIST_ROWS, kind="kde", **kw),
            "displot",
            True,
            "the kernel density estimate reads raw rows",
            _generic_bin_remedy("v"),
            "the kernel density estimate reads raw rows",
            _generic_aggregate_remedy("v"),
        ),
        "displot_ecdf": (
            lambda **kw: fm.displot(_HIST_ROWS, kind="ecdf", **kw),
            "displot",
            True,
            "the ECDF counts the raw values of x",
            _generic_bin_remedy("v"),
            "the ECDF counts the raw values of x",
            _generic_aggregate_remedy("v"),
        ),
        "lmplot": (
            lambda **kw: fm.lmplot(_XY_ROWS, y="w", **kw),
            "lmplot",
            True,
            "the fit reads the raw predictor",
            fit_bin_remedy,
            "the fit reads the raw predictor",
            _generic_aggregate_remedy("v"),
        ),
        "regplot": (
            lambda **kw: fm.regplot(_XY_ROWS, y="w", **kw),
            "regplot",
            True,
            "the fit reads the raw predictor",
            fit_bin_remedy,
            "the fit reads the raw predictor",
            _generic_aggregate_remedy("v"),
        ),
        "residplot": (
            lambda **kw: fm.residplot(_XY_ROWS, y="w", **kw),
            "residplot",
            True,
            "the residual fit reads the raw predictor and response",
            _generic_bin_remedy("v", call_kwargs="dropna=False"),
            "the residual fit reads the raw predictor and response",
            _generic_aggregate_remedy("v", call_kwargs="dropna=False"),
        ),
    }

    for door_id, (
        call,
        voice,
        bin_value,
        bin_reason,
        bin_remedy,
        agg_reason,
        agg_remedy,
    ) in doors.items():
        # Shorthand: carries only an aggregate.
        def build_shorthand(call=call):
            def run() -> None:
                call(x="mean(v)")

            return run

        cases.append(
            (
                f"{door_id}_shorthand_x_aggregate",
                build_shorthand(),
                _message(
                    voice,
                    _written("x", "X", "v", "aggregate", "mean"),
                    "aggregate",
                    agg_reason,
                    agg_remedy,
                ),
            )
        )

        # encode=: x bin=.
        def build_encode_bin(call=call, bin_value=bin_value):
            def run() -> None:
                call(x="v", encode={"x": X("v", bin=bin_value)})

            return run

        cases.append(
            (
                f"{door_id}_encode_x_bin",
                build_encode_bin(),
                _message(
                    voice, _written("x", "X", "v", "bin", bin_value), "bin", bin_reason, bin_remedy
                ),
            )
        )

        # encode=: x aggregate=.
        def build_encode_aggregate(call=call):
            def run() -> None:
                call(x="v", encode={"x": X("v", aggregate="mean")})

            return run

        cases.append(
            (
                f"{door_id}_encode_x_aggregate",
                build_encode_aggregate(),
                _message(
                    voice,
                    _written("x", "X", "v", "aggregate", "mean"),
                    "aggregate",
                    agg_reason,
                    agg_remedy,
                ),
            )
        )

    # displot hist with kde=True: the encode= reach survives the overlay
    # resolving the pending mark ahead of where encode= applies.
    cases.append(
        (
            "displot_hist_kde_overlay_encode_x_bin",
            lambda: fm.displot(
                _HIST_ROWS, x="v", kind="hist", kde=True, encode={"x": X("v", bin={"bin_count": 3})}
            ),
            _message(
                "displot",
                _written("x", "X", "v", "bin", {"bin_count": 3}),
                "bin",
                "the histogram bins x by its own rule",
                "State the rule with displot's bins=, or drop bin=.",
            ),
        )
    )
    return cases


#: Rows with a string hue column, for jointplot's/pairplot's share-mode
#: (normalized-stack) cases: the right marginal's/diagonal's own grouping
#: needs a real categorical field, not a plain float.
_XY_HUE_ROWS = _XY_ROWS.with_columns(g=pl.Series(["a", "b"] * 10))


def _revoiced_figure_cases() -> list[_Case]:
    """A re-voiced refusal's remedy, named in the voicing figure's own spelling.

    displot's and regplot's/lmplot's own keyword/``encode=``/post-hoc doors,
    and mark_density's/displot(kind='kde')'s post-hoc door, already resolve
    their own vocabulary through the generators above (:func:`_histogram_cases`,
    :func:`_kde_cases`). The cases here are the ones no generator's cell set
    covers: jointplot and pairplot, neither of which pre-checks at call time,
    so every one of their refusals -- keyword, ``encode=``, or a sub-panel
    (marginal/center/diagonal) ``encode=`` fans out to -- resolves through
    the same resolve-time refusal the generators exercise for their own doors.
    """
    cases: list[_Case] = []

    # jointplot: the marginal histogram's bin rule, reached as a keyword and
    # through encode= alike (jointplot has no pre-check of its own, so both
    # doors resolve through the same resolve-time refusal).
    for label, build_kwargs in (
        ("keyword", {"x": X("v", bin={"bin_count": 3}), "y": "w"}),
        ("encode", {"x": "v", "y": "w", "encode": {"x": X("v", bin={"bin_count": 3})}}),
    ):

        def build(build_kwargs=build_kwargs):
            def run() -> None:
                fm.jointplot(_XY_ROWS, **build_kwargs).to_svg()

            return run

        cases.append(
            (
                f"jointplot_hist_bin_rule_{label}",
                build(),
                _message(
                    "jointplot",
                    _written("x", "X", "v", "bin", {"bin_count": 3}),
                    "bin",
                    "the histogram bins x by its own rule",
                    "State the rule with marginal_kws={'bins': ...}, or drop bin=.",
                ),
            )
        )

    # jointplot: the top marginal's own binned axis, aggregate=.
    cases.append(
        (
            "jointplot_hist_x_aggregate",
            lambda: fm.jointplot(_XY_ROWS, x=X("v", aggregate="mean"), y="w").to_svg(),
            _message(
                "jointplot",
                _written("x", "X", "v", "aggregate", "mean"),
                "aggregate",
                "the histogram bins x rather than aggregating it",
                "Drop aggregate=.",
            ),
        )
    )

    # jointplot: the right marginal's value axis, field-bearing bin=/aggregate=.
    # jointplot's own encode= also reaches the right marginal's binned axis
    # (bound to the same "y" channel), so the drop clause cannot offer "style
    # the axis with a fieldless channel" the way every other door's does --
    # that alternative needs a field there too.
    cases.append(
        (
            "jointplot_hist_y_aggregate",
            lambda: fm.jointplot(
                _XY_ROWS, x="v", y="w", encode={"y": Y("w", aggregate="mean")}
            ).to_svg(),
            _message(
                "jointplot",
                _written("y", "Y", "w", "aggregate", "mean"),
                "aggregate",
                "the histogram computes y itself",
                "Draw the aggregate per bin with a bar mark, fm.Chart(data).mark_bar().encode("
                "x=fm.X('v', bin=True), y=fm.Y('w', aggregate='mean')), or drop y=.",
            ),
        )
    )
    cases.append(
        (
            "jointplot_hist_y_bin",
            lambda: fm.jointplot(_XY_ROWS, x="v", y="w", encode={"y": Y("w", bin=True)}).to_svg(),
            _message(
                "jointplot",
                _written("y", "Y", "w", "bin", True),
                "bin",
                "the histogram computes y itself",
                "Count rows over both fields with fm.Chart(data).mark_hex().encode(x='v', "
                "y='w'), or drop y=.",
            ),
        )
    )

    # jointplot: the top marginal's fieldless count under each value mode,
    # spelled through marginal_kws= (density) or marginal_kws=/the channel's
    # own stack= (the three share modes) -- every one falls back to "Drop
    # aggregate=.", since encode= reaching the top marginal's value axis
    # also reaches the right marginal's own binned axis (bound to the same
    # "y" channel), which refuses any aggregate unconditionally. See
    # :func:`test_jointplot_names_no_count_remedy_since_encode_reaches_its_binned_marginal`.
    for label, jointplot_kwargs, reason in (
        (
            "density",
            {"marginal_kws": {"density": True}},
            "with marginal_kws={'density': True} the histogram draws density on y, not a count",
        ),
        (
            "fill",
            {"hue": "g", "marginal_kws": {"multiple": "fill"}},
            "with marginal_kws={'multiple': 'fill'} the histogram draws each bin's share of "
            "its stack on y, not a count",
        ),
        (
            "position_normalize",
            {"hue": "g", "marginal_kws": {"position": fm.Stack(offset="normalize")}},
            "with marginal_kws={'position': fm.Stack(offset='normalize')} the histogram draws "
            "each bin's share of its stack on y, not a count",
        ),
        (
            "stack_normalize",
            {"hue": "g"},
            "with stack='normalize' the histogram draws each bin's share of its stack on y, "
            "not a count",
        ),
    ):
        y = (
            Y(aggregate="count", stack="normalize")
            if label == "stack_normalize"
            else Y(aggregate="count")
        )

        def build(jointplot_kwargs=jointplot_kwargs, y=y):
            def run() -> None:
                fm.jointplot(
                    _XY_HUE_ROWS, x="v", y="w", **jointplot_kwargs, encode={"y": y}
                ).to_svg()

            return run

        cases.append(
            (
                f"jointplot_hist_count_under_{label}",
                build(),
                _message(
                    "jointplot",
                    _written("y", "Y", None, "aggregate", "count"),
                    "aggregate",
                    reason,
                    "Drop aggregate=.",
                ),
            )
        )

    # jointplot: the center panel's own kernel density estimate, and its
    # kde marginal, both fall back to the knob-free remedy (no .transform()
    # on JointChart).
    cases.append(
        (
            "jointplot_kde_marginal_x_bin",
            lambda: fm.jointplot(
                _XY_ROWS, x="v", y="w", marginal_kind="kde", encode={"x": X("v", bin=True)}
            ).to_svg(),
            _message(
                "jointplot",
                _written("x", "X", "v", "bin", True),
                "bin",
                "the kernel density estimate reads raw rows",
                "Drop bin=.",
            ),
        )
    )
    cases.append(
        (
            "jointplot_kde_center_x_bin",
            lambda: fm.jointplot(
                _XY_ROWS, x="v", y="w", kind="kde", encode={"x": X("v", bin=True)}
            ).to_svg(),
            _message(
                "jointplot",
                _written("x", "X", "v", "bin", True),
                "bin",
                "the kernel density estimate reads raw rows",
                "Drop bin=.",
            ),
        )
    )

    # jointplot(kind="reg"): the fit's x bin= has no jointplot door of its
    # own (unlike mark_smooth/regplot/lmplot's x_bins=), so it falls back to
    # the knob-free remedy rather than naming a keyword jointplot lacks.
    cases.append(
        (
            "jointplot_reg_x_bin",
            lambda: fm.jointplot(
                _XY_ROWS, x="v", y="w", kind="reg", encode={"x": X("v", bin=True)}
            ).to_svg(),
            _message(
                "jointplot",
                _written("x", "X", "v", "bin", True),
                "bin",
                "the fit reads the raw predictor",
                "Drop bin=.",
            ),
        )
    )

    # pairplot: neither the fit's nor the kde's own knob has a pairplot
    # door (bare bin=True refuses on both, unlike the histogram's own bin
    # axis, which honors it as a redundancy), so each falls back to the
    # knob-free remedy. The diagonal is suppressed (diag_kind=None) so only
    # the off-diagonal panel's own mark refuses.
    for kind, reason in (
        ("reg", "the fit reads the raw predictor"),
        ("kde", "the kernel density estimate reads raw rows"),
    ):

        def build(kind=kind):
            def run() -> None:
                fm.pairplot(
                    _XY_ROWS,
                    kind=kind,
                    diag_kind=None,
                    vars=["v", "w"],
                    encode={"x": X("v", bin=True)},
                ).to_svg()

            return run

        cases.append(
            (
                f"pairplot_{kind}_x_bin",
                build(),
                _message(
                    "pairplot", _written("x", "X", "v", "bin", True), "bin", reason, "Drop bin=."
                ),
            )
        )

    cases.append(
        (
            "pairplot_hist_bin_rule",
            lambda: fm.pairplot(
                _XY_ROWS,
                kind="hist",
                diag_kind=None,
                vars=["v", "w"],
                encode={"x": X("v", bin={"bin_count": 3})},
            ).to_svg(),
            _message(
                "pairplot",
                _written("x", "X", "v", "bin", {"bin_count": 3}),
                "bin",
                "the histogram bins x by its own rule",
                "Drop bin=.",
            ),
        )
    )
    cases.append(
        (
            "pairplot_hist_x_aggregate",
            lambda: fm.pairplot(
                _XY_ROWS,
                kind="hist",
                diag_kind=None,
                vars=["v", "w"],
                encode={"x": X("v", aggregate="mean")},
            ).to_svg(),
            _message(
                "pairplot",
                _written("x", "X", "v", "aggregate", "mean"),
                "aggregate",
                "the histogram bins x rather than aggregating it",
                "Drop aggregate=.",
            ),
        )
    )

    # pairplot: the value axis, field-bearing bin=/aggregate= -- unlike
    # jointplot, pairplot's encode= reaches only the one off-diagonal
    # template, so the drop clause's "style with a fieldless channel"
    # alternative still renders here; byte-identical to the knob-free rows.
    cases.append(
        (
            "pairplot_hist_y_aggregate",
            lambda: fm.pairplot(
                _XY_ROWS,
                kind="hist",
                diag_kind=None,
                vars=["v", "w"],
                encode={"y": Y("w", aggregate="mean")},
            ).to_svg(),
            _message(
                "pairplot",
                _written("y", "Y", "w", "aggregate", "mean"),
                "aggregate",
                "the histogram computes y itself",
                "Draw the aggregate per bin with a bar mark, fm.Chart(data).mark_bar().encode("
                "x=fm.X('v', bin=True), y=fm.Y('w', aggregate='mean')), or drop y= (style the "
                "axis with fm.Y(title=...) and no field).",
            ),
        )
    )
    cases.append(
        (
            "pairplot_hist_y_bin",
            lambda: fm.pairplot(
                _XY_ROWS,
                kind="hist",
                diag_kind=None,
                vars=["v", "w"],
                encode={"y": Y("w", bin=True)},
            ).to_svg(),
            _message(
                "pairplot",
                _written("y", "Y", "w", "bin", True),
                "bin",
                "the histogram computes y itself",
                "Count rows over both fields with fm.Chart(data).mark_hex().encode(x='v', "
                "y='w'), or drop y= (style the axis with fm.Y(title=...) and no field).",
            ),
        )
    )
    cases.append(
        (
            "pairplot_hist_count_under_stack_normalize",
            lambda: fm.pairplot(
                _XY_HUE_ROWS,
                kind="hist",
                diag_kind=None,
                vars=["v", "w"],
                hue="g",
                encode={"y": Y(aggregate="count", stack="normalize")},
            ).to_svg(),
            _message(
                "pairplot",
                _written("y", "Y", None, "aggregate", "count"),
                "aggregate",
                "with stack='normalize' the histogram draws each bin's share of its stack on y, "
                "not a count",
                "Drop aggregate=, or draw the count with stack=True.",
            ),
        )
    )

    # pairplot's diagonal: off-diagonal kind="scatter" lowers bin= in place
    # without refusing (a primitive mark honors it), so these isolate the
    # diagonal's own kde (the "auto" default)/hist refusal.
    cases.append(
        (
            "pairplot_diag_kde_x_bin",
            lambda: fm.pairplot(
                _XY_ROWS, kind="scatter", vars=["v", "w"], encode={"x": X("v", bin=True)}
            ).to_svg(),
            _message(
                "pairplot",
                _written("x", "X", "v", "bin", True),
                "bin",
                "the kernel density estimate reads raw rows",
                "Drop bin=.",
            ),
        )
    )
    cases.append(
        (
            "pairplot_diag_hist_bin_rule",
            lambda: fm.pairplot(
                _XY_ROWS,
                kind="scatter",
                diag_kind="hist",
                vars=["v", "w"],
                encode={"x": X("v", bin={"bin_count": 3})},
            ).to_svg(),
            _message(
                "pairplot",
                _written("x", "X", "v", "bin", {"bin_count": 3}),
                "bin",
                "the histogram bins x by its own rule",
                "Drop bin=.",
            ),
        )
    )

    # jointplot(kind="hex"): its x=/y= reach both the hex center and the
    # matching marginal histogram. The marginal's own bin axis honors a
    # redundant bin=True, but the hex center never draws the 1-D bin it
    # would ask for, so the center's refusal is what speaks, naming
    # jointplot's own joint_kws spelling of mark_hex's bin_size=.
    for axis, channel, field, other, other_field in (
        ("x", "x", "v", "y", "w"),
        ("y", "y", "w", "x", "v"),
    ):

        def build(channel=channel, field=field, other=other, other_field=other_field):
            def run() -> None:
                ch = _axis_channel(channel, field, bin=True)
                fm.jointplot(_XY_ROWS, kind="hex", **{channel: ch, other: other_field}).to_svg()

            return run

        cases.append(
            (
                f"jointplot_hex_{axis}_bin_true_keyword",
                build(),
                _message(
                    "jointplot",
                    _written(channel, _CLS[channel], field, "bin", True),
                    "bin",
                    "the hexagons bin x and y by their own grid",
                    "Size the cells with joint_kws={'bin_size': ...}, or drop bin=.",
                ),
            )
        )

    cases.append(
        (
            "jointplot_hex_x_bin_true_encode",
            lambda: fm.jointplot(
                _XY_ROWS, x="v", y="w", kind="hex", encode={"x": X("v", bin=True)}
            ).to_svg(),
            _message(
                "jointplot",
                _written("x", "X", "v", "bin", True),
                "bin",
                "the hexagons bin x and y by their own grid",
                "Size the cells with joint_kws={'bin_size': ...}, or drop bin=.",
            ),
        )
    )
    cases.append(
        (
            "jointplot_hex_color_aggregate_mismatch",
            lambda: fm.jointplot(
                _XYZ_ROWS,
                x="v",
                y="w",
                kind="hex",
                joint_kws={"aggregate": "sum", "field": "z"},
                encode={"color": fm.Color("z", aggregate="mean")},
            ).to_svg(),
            _message(
                "jointplot",
                _written("color", "Color", "z", "aggregate", "mean"),
                "aggregate",
                "the cells are colored by the statistic's own value",
                "Aggregate the cells with joint_kws={'aggregate': 'mean', 'field': 'z'}, or "
                "drop aggregate=.",
            ),
        )
    )
    return cases


@pytest.mark.parametrize(
    "keep_kwargs,without_kwargs,y_count_stack",
    [
        pytest.param(
            {"marginal_kws": {"density": False}},
            {"marginal_kws": {"density": True}},
            False,
            id="density",
        ),
        pytest.param(
            {"hue": "g", "marginal_kws": {"multiple": "stack"}},
            {"hue": "g", "marginal_kws": {"multiple": "fill"}},
            False,
            id="fill",
        ),
        pytest.param(
            {"hue": "g", "marginal_kws": {"position": fm.Stack()}},
            {"hue": "g", "marginal_kws": {"position": fm.Stack(offset="normalize")}},
            False,
            id="position_normalize",
        ),
        pytest.param({"hue": "g"}, {"hue": "g"}, True, id="stack_normalize"),
    ],
)
def test_jointplot_names_no_count_remedy_since_encode_reaches_its_binned_marginal(
    keep_kwargs: dict, without_kwargs: dict, y_count_stack: bool
) -> None:
    """jointplot's ``encode=`` fans to both marginals (``ferrum._overrides``), so a
    fieldless count on the top marginal's value axis is also the right marginal's
    own binned-axis aggregate, which refuses any aggregate unconditionally. Every
    "keep the count" spelling that works at mark_histogram/displot therefore
    fails at jointplot too -- with the bin-axis reason, not the value-axis one --
    which is why its own remedy never names one (just "Drop aggregate="). Without
    the ``y=`` binding at all, the same call renders.
    """
    y = Y(aggregate="count", stack=True) if y_count_stack else Y(aggregate="count")
    with pytest.raises(ValueError) as excinfo:
        fm.jointplot(_XY_HUE_ROWS, x="v", y="w", **keep_kwargs, encode={"y": y}).to_svg()
    assert "the histogram bins y rather than aggregating it" in str(excinfo.value)

    fm.jointplot(_XY_HUE_ROWS, x="v", y="w", **without_kwargs).to_svg()


_ALL_CASES: list[_Case] = (
    _histogram_cases("mark_histogram")
    + _histogram_cases("displot")
    + _histogram_cases("displot_posthoc")
    + _smooth_cases()
    + _fit_cases()
    + _kde_cases()
    + _contour_cases()
    + _hex_cases()
    + _raster_cases()
    + _qq_cases()
    + _ecdf_cases()
    + _residplot_cases()
    + _mark_function_cases()
    + _spelling_cases()
    + _revoiced_figure_cases()
)


@pytest.mark.parametrize(
    "builder,expected",
    [case[1:] for case in _ALL_CASES],
    ids=[case[0] for case in _ALL_CASES],
)
def test_a_statistic_refuses_with_the_full_message(
    builder: Callable[[], Any], expected: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        builder()
    assert str(excinfo.value) == expected


# Every figure-voiced case in the grid above must never misquote an inner
# mark's name, an inner parameter the caller cannot write, or a fieldless
# channel's dropped field -- checked over every such case, not a curated
# subset.
_FIGURE_VOICES = frozenset({"displot", "residplot", "lmplot", "regplot", "jointplot", "pairplot"})
_FORBIDDEN_IN_A_FIGURE_MESSAGE = ("mark_histogram", "mark_density", "mark_smooth", "None,")
_FIGURE_CASES = [case for case in _ALL_CASES if case[2].split("(", 1)[0] in _FIGURE_VOICES]

#: ``scripts/audit_docstring_drift.py``'s ``_doc_params``, the independent
#: oracle for what a ``*_kws=`` dict a figure forwards verbatim accepts:
#: the NumPy ``Parameters`` names of the mark it forwards to, parsed from
#: its own docstring (``mark_histogram``'s own knobs are ``**kwargs``, so
#: its docstring is the only record of what ``marginal_kws`` accepts).
#: Loaded once, as a module (not a package import: ``scripts/`` is a
#: directory of plain scripts, not an installed package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_docstring_drift = load_script(
    "audit_docstring_drift", _REPO_ROOT / "scripts" / "audit_docstring_drift.py"
)

#: Which mark a figure's ``*_kws=`` dict forwards to, keyed by
#: ``(voice, kws_name)``: only the histogram refusal names ``marginal_kws``
#: keys. A door-vocabulary row that starts naming a ``*_kws=`` dict this
#: table has no entry for fails the membership assertion below --
#: deliberately, since the only way to pass is to add the row here, next
#: to the mark it actually forwards to.
_KWS_FORWARDS_TO: dict[tuple[str, str], Callable[..., fm.Chart]] = {
    ("jointplot", "marginal_kws"): fm.Chart.mark_histogram,
    # jointplot forwards joint_kws to the center mark verbatim; only the hex
    # center's refusal (kind="hex") ever names a joint_kws key.
    ("jointplot", "joint_kws"): fm.Chart.mark_hex,
}

#: A call token: an identifier immediately before ``(`` (pushed onto the
#: enclosing-call stack), a bare ``(`` (prose or an unnamed call, pushed as
#: ``""``), a closing ``)`` (popped), or a bare ``name=`` kwarg token.
_CALL_TOKEN = re.compile(r"([A-Za-z_]\w*)?\(|\)|\b([a-z_]\w*)=")
#: A ``*_kws={...}`` spelling and its dict body.
_KWS_DICT = re.compile(r"\b([a-z_]+_kws)=\{([^}]*)\}")
#: A ``'key':`` entry inside a dict body.
_DICT_KEY = re.compile(r"'([a-z_]+)':")


def _assert_the_remedy_is_typeable_at_the_voice(message: str) -> None:
    """Every keyword, ``*_kws=`` key and chained method *message*'s reason/remedy
    half names is real at the voicing figure -- checked against independent
    oracles (signatures, channel-honored kwargs, the forwarded mark's own
    documented ``Parameters``, the figure's return annotation), never the
    production door-vocabulary table itself, so a wrong table entry fails
    here instead of agreeing with itself. The written channel half (before
    ``") is not supported: "``) is the caller's own and is not checked.
    """
    voice = message.split("(", 1)[0]
    fn = getattr(fm, voice, None)
    if fn is None:
        return
    figure_params = set(inspect.signature(fn).parameters)
    channel_kwargs = X._honored_kwargs | Y._honored_kwargs
    channel_names = {X._channel_name, Y._channel_name}
    _, _, rest = message.partition(") is not supported: ")

    stack: list[str] = []
    for match in _CALL_TOKEN.finditer(rest):
        call_name, kwarg = match.groups()
        if match.group(0) == ")":
            if stack:
                stack.pop()
            continue
        if kwarg is None:
            stack.append(call_name or "")
            continue
        inner = stack[-1] if stack else ""
        if inner == "Bin":
            allowed = set(inspect.signature(fm.Bin).parameters)
        elif inner == "Stack":
            allowed = set(inspect.signature(fm.Stack).parameters)
        elif inner in ("X", "Y"):
            allowed = channel_kwargs
        elif inner == "encode":
            allowed = channel_names
        else:
            allowed = figure_params | channel_kwargs | channel_names
        assert kwarg in allowed, (voice, kwarg, message)

    for kws_name, dict_body in _KWS_DICT.findall(rest):
        assert (voice, kws_name) in _KWS_FORWARDS_TO, (voice, kws_name, message)
        target = _KWS_FORWARDS_TO[(voice, kws_name)]
        documented = _docstring_drift._doc_params(inspect.getdoc(target)) or []
        for key in _DICT_KEY.findall(dict_body):
            assert key in documented, (voice, kws_name, key, message)

    if ".transform(" in rest:
        return_type = inspect.signature(fn, eval_str=True).return_annotation
        assert hasattr(return_type, "transform"), (voice, message)


@pytest.mark.parametrize(
    "builder,expected",
    [case[1:] for case in _FIGURE_CASES],
    ids=[case[0] for case in _FIGURE_CASES],
)
def test_a_figure_doors_refusal_never_misquotes_an_inner_call(
    builder: Callable[[], Any], expected: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        builder()
    message = str(excinfo.value)
    for word in _FORBIDDEN_IN_A_FIGURE_MESSAGE:
        assert word not in message, (word, message)
    if message.startswith("displot("):
        assert "density=True" not in message, message
    _assert_the_remedy_is_typeable_at_the_voice(message)


@pytest.mark.parametrize(
    "message",
    [
        "jointplot(x=X('v', bin=True)) is not supported: the histogram bins x by its own "
        "rule, so bin= would be ignored. Drop aggregate=, or draw the count with "
        "multiple='stack'.",
        "jointplot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins x "
        "by its own rule, so bin= would be ignored. State the rule with mark_histogram's "
        "bins=, bin_count= or bin_width=, or drop bin=.",
        "jointplot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins x "
        "by its own rule, so bin= would be ignored. State the rule with "
        "marginal_kws={'binz': ...}, or drop bin=.",
        "pairplot(x=X('v', bin=True)) is not supported: the fit reads the raw predictor, "
        "so bin= would be ignored. Bin the field per row with "
        ".transform(fm.Bin('v', per_row=True)) and bind its 'bin_start' column, or drop bin=.",
        "displot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins x by "
        "its own rule, so bin= would be ignored. State the rule with displot's binwidth=, "
        "or drop bin=.",
        "regplot(x=X('v', bin=True)) is not supported: the fit reads the raw predictor, so "
        "bin= would be ignored. State the rule with marginal_kws={'bins': ...}, or drop bin=.",
        "jointplot(x=X('v', bin=True)) is not supported: the fit reads the raw predictor, "
        "so bin= would be ignored. Bin the predictor with x_bins=<number of bins>, the "
        "fit's own binning, or drop bin=.",
        "jointplot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins x "
        "by its own rule, so bin= would be ignored. State the rule with "
        "joint_kws={'bins': ...}, or drop bin=.",
        "jointplot(x=X('v', bin=True)) is not supported: the hexagons bin x and y by "
        "their own grid, so bin= would be ignored. Size the cells with mark_hex's "
        "bin_size=, or drop bin=.",
        "pairplot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins x "
        "by its own rule, so bin= would be ignored. State the rule with "
        "joint_kws={'bins': ...}, or drop bin=.",
    ],
    ids=[
        "sibling_door_multiple",
        "sibling_door_bins",
        "unmapped_marginal_kws_key",
        "untypeable_transform",
        "misspelled_bins",
        "unmapped_marginal_kws_voice",
        "unowned_x_bins",
        "undocumented_joint_kws_key",
        "hex_center_knob_at_jointplot",
        "joint_kws_at_pairplot",
    ],
)
def test_the_typeable_remedy_check_rejects_a_misquote(message: str) -> None:
    """Every synthetic misquote above raises; a real figure message does not."""
    with pytest.raises(AssertionError):
        _assert_the_remedy_is_typeable_at_the_voice(message)


def test_the_typeable_remedy_check_accepts_a_real_jointplot_message() -> None:
    """Positive controls: the actual jointplot messages pass untouched."""
    _assert_the_remedy_is_typeable_at_the_voice(
        "jointplot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins x "
        "by its own rule, so bin= would be ignored. State the rule with "
        "marginal_kws={'bins': ...}, or drop bin=."
    )
    _assert_the_remedy_is_typeable_at_the_voice(
        "jointplot(x=X('v', bin=True)) is not supported: the hexagons bin x and y by "
        "their own grid, so bin= would be ignored. Size the cells with "
        "joint_kws={'bin_size': ...}, or drop bin=."
    )
    _assert_the_remedy_is_typeable_at_the_voice(
        "jointplot(color=Color('z', aggregate='mean')) is not supported: the cells are "
        "colored by the statistic's own value, so aggregate= would be ignored. Aggregate "
        "the cells with joint_kws={'aggregate': 'mean', 'field': 'z'}, or drop aggregate=."
    )


@pytest.mark.parametrize(
    "refused,voice,expected",
    [
        (
            lambda: (
                fm.Chart(_HIST_ROWS)
                .mark_histogram(density=True)
                .encode(x="v", y=Y(aggregate="count"))
                .to_svg()
            ),
            "pairplot",
            "pairplot(y=Y(aggregate='count')) is not supported: the histogram computes y "
            "itself, so aggregate= would be ignored. Drop aggregate=.",
        ),
        (
            lambda: (
                fm.Chart(_HIST_ROWS)
                .mark_histogram(density=True)
                .encode(x="v", y=Y(aggregate="count"))
                .to_svg()
            ),
            "roc_chart",
            "roc_chart(y=Y(aggregate='count')) is not supported: the histogram computes y "
            "itself, so aggregate= would be ignored. Drop aggregate=.",
        ),
        (
            lambda: (
                fm.Chart(_HIST_ROWS)
                .mark_histogram()
                .encode(x=X("v", bin={"bin_count": 3}))
                .to_svg()
            ),
            "pairplot",
            "pairplot(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins "
            "x by its own rule, so bin= would be ignored. Drop bin=.",
        ),
        (
            lambda: (
                fm.Chart(_HIST_ROWS)
                .mark_histogram()
                .encode(x=X("v", bin={"bin_count": 3}))
                .to_svg()
            ),
            "roc_chart",
            "roc_chart(x=X('v', bin={'bin_count': 3})) is not supported: the histogram bins "
            "x by its own rule, so bin= would be ignored. Bin the field per row with "
            ".transform(fm.Bin('v', per_row=True)) and bind its 'bin_start' column, or drop "
            "bin=.",
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_smooth().encode(x=X("v", bin=True), y="w").to_svg(),
            "jointplot",
            "jointplot(x=X('v', bin=True)) is not supported: the fit reads the raw "
            "predictor, so bin= would be ignored. Drop bin=.",
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_smooth().encode(x=X("v", bin=True), y="w").to_svg(),
            "residplot",
            "residplot(x=X('v', bin=True)) is not supported: the fit reads the raw "
            "predictor, so bin= would be ignored. Bin the field per row with "
            ".transform(fm.Bin('v', per_row=True)) and bind its 'bin_start' column, or drop "
            "bin=.",
        ),
        (
            lambda: (
                fm.Chart(_GROUPED_HIST_ROWS)
                .mark_histogram(multiple="fill", groupby=["g"])
                .encode(x="v", color="g", y=Y(aggregate="count"))
                .to_svg()
            ),
            "roc_chart",
            "roc_chart(y=Y(aggregate='count')) is not supported: the histogram computes y "
            "itself, so aggregate= would be ignored. Drop aggregate=.",
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x=X("v", bin=True), y="w").to_svg(),
            "pairplot",
            "pairplot(x=X('v', bin=True)) is not supported: the hexagons bin x and y by "
            "their own grid, so bin= would be ignored. Drop bin=.",
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x=X("v", bin=True), y="w").to_svg(),
            "jointplot",
            "jointplot(x=X('v', bin=True)) is not supported: the raster bins x and y onto "
            "its own pixel grid, so bin= would be ignored. Drop bin=.",
        ),
        (
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_contour()
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            "jointplot",
            "jointplot(color=Color('z', aggregate='mean')) is not supported: the levels are "
            "colored by the statistic's own value, so aggregate= would be ignored. Drop "
            "aggregate=.",
        ),
        (
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(aggregate="sum", field="z")
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            "roc_chart",
            "roc_chart(color=Color('z', aggregate='mean')) is not supported: the cells are "
            "colored by the statistic's own value, so aggregate= would be ignored. Drop "
            "aggregate=.",
        ),
        (
            lambda: fm.Chart(_HIST_ROWS).mark_histogram().encode(x=X("v", type="N")).to_svg(),
            "pairplot",
            "pairplot(x=X('v', type='N')) is not supported: the histogram bins x as numbers "
            "and draws a bar per bin, so type='N' would draw one bar per row instead of one "
            "per category. Drop the type (x='v').",
        ),
    ],
    ids=[
        "mark_histogram_density_count_as_pairplot",
        "mark_histogram_density_count_as_roc_chart",
        "mark_histogram_bin_rule_as_pairplot",
        "mark_histogram_bin_rule_as_roc_chart",
        "mark_smooth_x_bin_as_jointplot",
        "mark_smooth_x_bin_as_residplot",
        "mark_histogram_fill_count_as_roc_chart",
        "mark_hex_x_bin_as_pairplot",
        "mark_raster_x_bin_as_jointplot",
        "mark_contour_color_aggregate_as_jointplot",
        "mark_hex_color_mismatch_as_roc_chart",
        "mark_histogram_categorical_type_as_pairplot",
    ],
)
def test_a_voice_without_the_knob_takes_the_knob_free_remedy(
    refused: Callable[[], Any], voice: str, expected: str
) -> None:
    """A voice whose row lacks the knob (pairplot, jointplot) or is absent from the
    table entirely (roc_chart) re-voices with the row's own ``None`` fallback --
    the knob-free remedy, never a sibling door's spelling.
    """
    with pytest.raises(ValueError) as excinfo:
        refused()
    assert str(excinfo.value.spoken_as(voice)) == expected


#: Every door this family's refusals re-voice into, plus a voice absent
#: from the table (``roc_chart``), for the pickle/copy contract below.
_PICKLE_CONTRACT_DOORS = (
    "mark_histogram",
    "displot",
    "jointplot",
    "pairplot",
    "mark_smooth",
    "regplot",
    "lmplot",
    "roc_chart",
    "mark_hex",
    "mark_raster",
    "mark_contour",
)
#: One refusal per door-spelled knob, built directly from the public API
#: (no ``_encode_lowering`` import), each a case
#: ``EntryRefusal.__reduce__``'s slot-substitution premise does not hold
#: for: the wording itself, not only the voice, differs per door.
_PICKLE_CONTRACT_REFUSALS: dict[str, Callable[[], Any]] = {
    "bin_rule": lambda: (
        fm.Chart(_HIST_ROWS).mark_histogram().encode(x=X("v", bin={"bin_count": 3})).to_svg()
    ),
    "density_posthoc": lambda: (
        fm.displot(_HIST_ROWS, x="v", kind="hist", stat="density")
        .encode(y=Y(aggregate="count"))
        .to_svg()
    ),
    "fit_x_bins": lambda: (
        fm.Chart(_XY_ROWS).mark_smooth().encode(x=X("v", bin=True), y="w").to_svg()
    ),
    "hex_bin_size": lambda: (
        fm.Chart(_XY_ROWS).mark_hex().encode(x=X("v", bin=True), y="w").to_svg()
    ),
    "hex_cells": lambda: (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(aggregate="sum", field="z")
        .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
        .to_svg()
    ),
    "categorical_histogram": lambda: (
        fm.Chart(_HIST_ROWS).mark_histogram().encode(x=X("v", type="N")).to_svg()
    ),
}


def _refused(builder: Callable[[], Any]) -> Any:
    with pytest.raises(ValueError) as excinfo:
        builder()
    return excinfo.value


def _pickle_copy(value: Any) -> Any:
    # Round-trips an EntryRefusal this same test process just raised --
    # never data from an external or untrusted source -- the same
    # in-process pickle/unpickle `tests/test_public_entry_voice.py:765`
    # exercises for the "a render in a process pool sends its refusal back
    # pickled" contract.
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize(
    "copy_fn",
    [_pickle_copy, copy.copy, copy.deepcopy],
    ids=["pickle", "copy", "deepcopy"],
)
@pytest.mark.parametrize(
    "refusal_id",
    [*_PICKLE_CONTRACT_REFUSALS, "prefixed"],
)
def test_a_door_spelled_refusal_survives_pickle_and_copy(
    refusal_id: str, copy_fn: Callable[[Any], Any]
) -> None:
    """A door-spelled refusal keeps its exact text through pickle/copy/deepcopy,
    and a copy of that copy does too: ``EntryRefusal.__reduce__``'s own
    slot-substitution premise (the voice appears only as a call-name
    substring) does not hold for this family, since the wording itself
    differs per door, so the clone must freeze each declared door's whole
    text rather than rebuild it from one template. The tails re-voiced
    across every door are asserted NOT all equal, proving this test can
    tell door spellings apart -- a defect that collapsed every re-voicing
    to one fallback text would otherwise still pass trivially.
    """
    if refusal_id == "prefixed":
        original = _refused(_PICKLE_CONTRACT_REFUSALS["bin_rule"]).prefixed("compare={'m2': ...}: ")
    else:
        original = _refused(_PICKLE_CONTRACT_REFUSALS[refusal_id])

    clone = copy_fn(original)
    clone_of_clone = copy_fn(clone)

    tails = set()
    for door in _PICKLE_CONTRACT_DOORS:
        original_text = str(original.spoken_as(door))
        assert str(clone.spoken_as(door)) == original_text
        assert str(clone_of_clone.spoken_as(door)) == original_text
        tails.add(original_text.partition(") is not supported: ")[2])
    assert len(tails) > 1, "every door produced the same tail; cannot distinguish spellings"


#: Degenerate x/y frames, built the same way as the fit's own degenerate
#: cases (``tests/test_log_axis_statistics.py``): empty, a single row,
#: every row sharing one x value, and a null in each column.
_DEGENERATE_FRAMES: dict[str, pl.DataFrame] = {
    "empty": pl.DataFrame(
        {"v": pl.Series([], dtype=pl.Float64), "w": pl.Series([], dtype=pl.Float64)}
    ),
    "one_row": pl.DataFrame({"v": [5.0], "w": [5.0]}),
    "all_tied": pl.DataFrame({"v": [5.0, 5.0, 5.0, 5.0], "w": [1.0, 2.0, 3.0, 4.0]}),
    "nulls": pl.DataFrame({"v": [1.0, None, 3.0], "w": [1.0, 2.0, None]}),
}

#: One re-voicing builder per door/route this family covers, each over a
#: ``rows`` frame it reads only for its own x/y columns -- never, in any
#: of them, for the refusal itself.
_REVOICED_BUILDERS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "displot_posthoc_bin_rule": lambda rows: fm.displot(rows, x="v", kind="hist").encode(
        x=X("v", bin={"bin_count": 3})
    ),
    "displot_posthoc_stat_density": lambda rows: fm.displot(
        rows, x="v", kind="hist", stat="density"
    ).encode(y=Y(aggregate="count")),
    "displot_posthoc_kde_x_bin": lambda rows: fm.displot(rows, x="v", kind="kde").encode(
        x=X("v", bin=True)
    ),
    "jointplot_bin_rule_keyword": lambda rows: fm.jointplot(
        rows, x=X("v", bin={"bin_count": 3}), y="w"
    ),
    "jointplot_density": lambda rows: fm.jointplot(
        rows, x="v", y="w", marginal_kws={"density": True}, encode={"y": Y(aggregate="count")}
    ),
    "jointplot_reg_x_bin": lambda rows: fm.jointplot(
        rows, x="v", y="w", kind="reg", encode={"x": X("v", bin=True)}
    ),
    "pairplot_hist_bin_rule": lambda rows: fm.pairplot(
        rows,
        kind="hist",
        diag_kind=None,
        vars=["v", "w"],
        encode={"x": X("v", bin={"bin_count": 3})},
    ),
    "pairplot_reg_x_bin": lambda rows: fm.pairplot(
        rows, kind="reg", diag_kind=None, vars=["v", "w"], encode={"x": X("v", bin=True)}
    ),
    "regplot_posthoc_x_bin": lambda rows: fm.regplot(rows, x="v", y="w", scatter=False).encode(
        x=X("v", bin=True)
    ),
    "lmplot_posthoc_x_bin": lambda rows: fm.lmplot(rows, x="v", y="w", scatter=False).encode(
        x=X("v", bin=True)
    ),
    "jointplot_hex_x_bin_true": lambda rows: fm.jointplot(
        rows, x=X("v", bin=True), y="w", kind="hex"
    ),
    "jointplot_hex_color_mismatch": lambda rows: fm.jointplot(
        rows.with_columns(z=pl.col("w")),
        x="v",
        y="w",
        kind="hex",
        joint_kws={"aggregate": "sum", "field": "z"},
        encode={"color": fm.Color("z", aggregate="mean")},
    ),
}


@pytest.mark.parametrize("builder_id", list(_REVOICED_BUILDERS))
@pytest.mark.parametrize("frame_id", list(_DEGENERATE_FRAMES))
def test_a_re_voiced_refusal_reads_no_rows(builder_id: str, frame_id: str) -> None:
    """Every re-voiced refusal in this family is raised from the encoding
    alone, never the rows: an empty, one-row, all-tied or null frame
    refuses with the exact same message a full frame does.
    """
    builder = _REVOICED_BUILDERS[builder_id]

    def message_for(rows: pl.DataFrame) -> str:
        with pytest.raises(ValueError) as excinfo:
            builder(rows).to_svg()
        return str(excinfo.value)

    assert message_for(_DEGENERATE_FRAMES[frame_id]) == message_for(_XY_ROWS)


#: One builder per direct refusal of the four statistic marks (contour,
#: hex, raster, qq): their x/y bin=/aggregate=, their color refusals, and
#: mark_qq's field-bearing y, mirroring :data:`_REVOICED_BUILDERS` by
#: reference: each a ``rows -> Chart`` callable, read only for its own
#: x/y/color columns, never for the refusal itself. A color row derives
#: ``z`` from ``w`` so every frame below (including the degenerate ones,
#: which carry no ``z``) has one.
_STATISTIC_MARK_BUILDERS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "mark_contour_x_bin": lambda rows: (
        fm.Chart(rows).mark_contour().encode(x=X("v", bin=True), y="w")
    ),
    "mark_contour_fill_y_aggregate": lambda rows: (
        fm.Chart(rows).mark_contour(fill=True).encode(x="v", y=Y("w", aggregate="mean"))
    ),
    "mark_contour_color": lambda rows: (
        fm.Chart(rows.with_columns(z=pl.col("w")))
        .mark_contour()
        .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
    ),
    "mark_hex_x_bin": lambda rows: fm.Chart(rows).mark_hex().encode(x=X("v", bin=True), y="w"),
    "mark_hex_y_aggregate": lambda rows: (
        fm.Chart(rows).mark_hex().encode(x="v", y=Y("w", aggregate="mean"))
    ),
    "mark_hex_color_mismatch": lambda rows: (
        fm.Chart(rows.with_columns(z=pl.col("w")))
        .mark_hex(aggregate="sum", field="z")
        .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
    ),
    "mark_raster_x_bin": lambda rows: (
        fm.Chart(rows).mark_raster().encode(x=X("v", bin=True), y="w")
    ),
    "mark_raster_color": lambda rows: (
        fm.Chart(rows.with_columns(z=pl.col("w")))
        .mark_raster()
        .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
    ),
    "mark_qq_x_bin": lambda rows: fm.Chart(rows).mark_qq().encode(x=X("v", bin=True)),
    "mark_qq_y": lambda rows: fm.Chart(rows).mark_qq().encode(x="v", y="w"),
    "mark_qq_fieldless_y_aggregate": lambda rows: (
        fm.Chart(rows).mark_qq().encode(x="v", y=Y(aggregate="mean"))
    ),
}


@pytest.mark.parametrize("builder_id", list(_STATISTIC_MARK_BUILDERS))
@pytest.mark.parametrize("frame_id", list(_DEGENERATE_FRAMES))
def test_a_statistic_mark_refusal_reads_no_rows(builder_id: str, frame_id: str) -> None:
    """Every direct refusal of the four statistic marks is raised from the
    encoding alone, never the rows: an empty, one-row, all-tied or null
    frame refuses with the exact same message a full frame does.
    """
    builder = _STATISTIC_MARK_BUILDERS[builder_id]

    def message_for(rows: pl.DataFrame) -> str:
        with pytest.raises(ValueError) as excinfo:
            builder(rows).to_svg()
        return str(excinfo.value)

    assert message_for(_DEGENERATE_FRAMES[frame_id]) == message_for(_XY_ROWS)


def test_ecdf_refuses_a_field_bearing_y_reached_only_through_encode():
    """``encode={'y': 'v'}`` draws the same field-bearing-y refusal a keyword ``y='v'`` gets."""
    with pytest.raises(ValueError, match="kind='ecdf'"):
        fm.displot(_HIST_ROWS, x="v", kind="ecdf", encode={"y": "v"})


#: The one validator's message for an unknown ``multiple=`` on mark_histogram.
_UNKNOWN_MULTIPLE = (
    "mark_histogram: multiple must be one of ['dodge', 'fill', 'layer', 'stack']; got 'bogus'"
)


@pytest.mark.parametrize(
    "mark_kwargs,encode",
    [
        ({}, {"x": "v"}),
        ({}, {"x": "v", "y": Y(aggregate="count")}),
        ({"orientation": "horizontal"}, {"y": "v", "x": X(aggregate="count")}),
        (
            {"groupby": ["g"], "position": fm.Stack(offset="normalize")},
            {"x": "v", "color": "g", "y": Y(aggregate="count")},
        ),
        (
            {"groupby": ["g"]},
            {"x": "v", "color": "g", "y": Y(aggregate="count", stack="normalize")},
        ),
    ],
    ids=["plain", "fieldless_count", "horizontal_count", "normalize_position", "normalize_stack"],
)
def test_mark_histogram_refuses_an_unknown_multiple_in_its_own_voice(
    mark_kwargs: dict, encode: dict
):
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_GROUPED_HIST_ROWS).mark_histogram(multiple="bogus", **mark_kwargs).encode(
            **encode
        ).to_svg()
    assert str(excinfo.value) == _UNKNOWN_MULTIPLE


# ---------------------------------------------------------------------------
# Remedy runs: each remedy a refusal names is run on the door that names it,
# as its message writes it, and compared with a sibling route that draws the
# same thing another way. Each run also shows the call failing with the
# remedy's own step left out, so a remedy that leads nowhere cannot pass.
# ---------------------------------------------------------------------------

#: A stated per-row rule, so polars can bin a sibling frame the same way:
#: ``bin_start = (field // 5) * 5`` over ``[0, 45)``, which covers ``v``
#: (1..20) and ``w`` (3..41) alike.
_PER_ROW_RULE: dict[str, Any] = {"bin_width": 5.0, "extent": (0.0, 45.0), "nice": False}

#: Rows whose binary response varies within every bin, so a logistic fit
#: over the binned predictor is not separable.
_NONSEPARABLE_ROWS = _XY_ROWS.with_columns(b=((pl.col("v") % 3) == 0).cast(pl.Int64))

#: Rows in five groups of three, for the ``transform_aggregate`` remedy.
_GROUPED_XY_ROWS = pl.DataFrame(
    {
        "g": [i // 3 for i in range(15)],
        "v": [float(i + 1) for i in range(15)],
        "w": [1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 4.0, 7.0, 8.0, 6.0, 8.0, 10.0, 7.0, 9.0, 13.0],
    }
)
#: ``_GROUPED_XY_ROWS`` aggregated in polars: each group's mean ``v`` and ``w``.
_PREAGGREGATED_ROWS = (
    _GROUPED_XY_ROWS.group_by("g", maintain_order=True)
    .agg(mean_v=pl.col("v").mean(), mean_w=pl.col("w").mean())
    .sort("g")
)

#: ``_GROUPED_XY_ROWS`` with a hue level ``h`` crossed with the groups.
_GROUPED_HUE_ROWS = _GROUPED_XY_ROWS.with_columns(h=pl.col("g") % 2)
#: ``_GROUPED_HUE_ROWS`` aggregated in polars: each (g, h) group's mean ``v``.
_PREAGGREGATED_HUE_ROWS = (
    _GROUPED_HUE_ROWS.group_by(["g", "h"], maintain_order=True)
    .agg(mean_v=pl.col("v").mean())
    .sort("g")
)


def _per_group_and_hue_mean() -> dict:
    """The ``transform_aggregate`` remedy step under a hue: each (g, h) group's mean ``v``."""
    return fm.transform_aggregate({"field": "v", "fn": "mean", "as": "mean_v"}, groupby=["g", "h"])


_BIN_REMEDY_TEXT = (
    ".transform(fm.Bin({field!r}, per_row=True)) and bind its 'bin_start' column{with_clause}, "
    "or drop bin=."
)
_AGGREGATE_REMEDY_TEXT = (
    ".transform(fm.transform_aggregate(...)) and bind its output column{with_clause}, "
    "or drop aggregate=."
)


def _names_bin(field: str, with_clause: str = "") -> str:
    return _BIN_REMEDY_TEXT.format(field=field, with_clause=with_clause)


def _names_aggregate(with_clause: str = "") -> str:
    return _AGGREGATE_REMEDY_TEXT.format(with_clause=with_clause)


def _per_row_bin(field: str) -> fm.Bin:
    """The remedy's own ``fm.Bin(field, per_row=True)``, with the stated rule."""
    return fm.Bin(field, per_row=True, **_PER_ROW_RULE)


def _prebinned(rows: pl.DataFrame, field: str) -> pl.DataFrame:
    """*rows* with the ``bin_start`` column polars computes under the same rule."""
    return rows.with_columns(bin_start=(pl.col(field) // 5.0) * 5.0)


def _per_group_mean() -> dict:
    """The ``transform_aggregate`` remedy step: each group's mean ``v`` and ``w``."""
    return fm.transform_aggregate(
        {"field": "v", "fn": "mean", "as": "mean_v"},
        {"field": "w", "fn": "mean", "as": "mean_w"},
        groupby=["g"],
    )


@dataclass(frozen=True)
class _MissingColumn:
    """A generic ``.transform(...)`` remedy's own step left out.

    *build* is the remedied call with that step removed: it fails because
    *column*, the fresh output the step would have bound, is never
    computed.
    """

    build: Callable[[], fm.Chart]
    error: type[Exception]
    column: str


@dataclass(frozen=True)
class _DifferentChart:
    """A named-keyword remedy's own keyword left out.

    There is no output column to leave uncomputed -- the keyword switches
    the voicing figure's own setting, not a transform step -- so *build*,
    the same call without that keyword, renders a DIFFERENT chart instead
    of failing.
    """

    build: Callable[[], fm.Chart]


@dataclass(frozen=True)
class _RemedyRun:
    """One remedy, run on the door whose refusal names it.

    *refused* raises the refusal, whose message must carry *names*.
    *remedied* is the same door with the remedy applied as the message
    writes it, and *sibling* is the same door drawing the same thing a
    different way. *without* is the required negative control, proving
    the remedy's own step actually matters: :class:`_MissingColumn` for a
    generic remedy (the step left out fails, naming the column it would
    have bound) or :class:`_DifferentChart` for a named-keyword remedy
    (the keyword left out renders, but not what *remedied* renders).
    """

    refused: Callable[[], Any]
    names: str
    remedied: Callable[[], fm.Chart]
    sibling: Callable[[], fm.Chart]
    without: _MissingColumn | _DifferentChart


_COLUMN_NOT_FOUND = pl.exceptions.ColumnNotFoundError

_REMEDY_RUNS: dict[str, _RemedyRun] = {
    # The per-row Bin: bin the refused field per row, bind 'bin_start'.
    "per_row_bin_regplot_logistic_x": _RemedyRun(
        refused=lambda: fm.regplot(
            _NONSEPARABLE_ROWS, x=X("v", bin=True), y="b", method="logistic"
        ),
        names=_names_bin("v"),
        remedied=lambda: fm.regplot(
            _NONSEPARABLE_ROWS, x="bin_start", y="b", method="logistic"
        ).transform(_per_row_bin("v")),
        sibling=lambda: fm.regplot(
            _prebinned(_NONSEPARABLE_ROWS, "v"), x="bin_start", y="b", method="logistic"
        ),
        without=_MissingColumn(
            build=lambda: fm.regplot(_NONSEPARABLE_ROWS, x="bin_start", y="b", method="logistic"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_regplot_robust_x": _RemedyRun(
        refused=lambda: fm.regplot(_XY_ROWS, x=X("v", bin=True), y="w", method="robust"),
        names=_names_bin("v"),
        remedied=lambda: fm.regplot(_XY_ROWS, x="bin_start", y="w", method="robust").transform(
            _per_row_bin("v")
        ),
        sibling=lambda: fm.regplot(
            _prebinned(_XY_ROWS, "v"), x="bin_start", y="w", method="robust"
        ),
        without=_MissingColumn(
            build=lambda: fm.regplot(_XY_ROWS, x="bin_start", y="w", method="robust"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_lmplot_lm_y": _RemedyRun(
        refused=lambda: fm.lmplot(_XY_ROWS, x="v", y=Y("w", bin=True), method="lm"),
        names=_names_bin("w"),
        remedied=lambda: fm.lmplot(_XY_ROWS, x="v", y="bin_start", method="lm").transform(
            _per_row_bin("w")
        ),
        sibling=lambda: fm.lmplot(_prebinned(_XY_ROWS, "w"), x="v", y="bin_start", method="lm"),
        without=_MissingColumn(
            build=lambda: fm.lmplot(_XY_ROWS, x="v", y="bin_start", method="lm"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_mark_density_x": _RemedyRun(
        refused=lambda: fm.Chart(_XY_ROWS).mark_density().encode(x=X("v", bin=True)).to_svg(),
        names=_names_bin("v"),
        remedied=lambda: (
            fm.Chart(_XY_ROWS).transform(_per_row_bin("v")).mark_density().encode(x="bin_start")
        ),
        sibling=lambda: fm.Chart(_prebinned(_XY_ROWS, "v")).mark_density().encode(x="bin_start"),
        without=_MissingColumn(
            build=lambda: fm.Chart(_XY_ROWS).mark_density().encode(x="bin_start"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_mark_density_bivariate_y": _RemedyRun(
        refused=lambda: (
            fm.Chart(_XY_ROWS).mark_density().encode(x="v", y=Y("w", bin=True)).to_svg()
        ),
        names=_names_bin("w"),
        remedied=lambda: (
            fm.Chart(_XY_ROWS)
            .transform(_per_row_bin("w"))
            .mark_density()
            .encode(x="v", y="bin_start")
        ),
        sibling=lambda: (
            fm.Chart(_prebinned(_XY_ROWS, "w")).mark_density().encode(x="v", y="bin_start")
        ),
        without=_MissingColumn(
            build=lambda: fm.Chart(_XY_ROWS).mark_density().encode(x="v", y="bin_start"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_displot_kde_x": _RemedyRun(
        refused=lambda: fm.displot(_XY_ROWS, x=X("v", bin=True), kind="kde"),
        names=_names_bin("v"),
        remedied=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="kde").transform(
            _per_row_bin("v")
        ),
        sibling=lambda: fm.displot(_prebinned(_XY_ROWS, "v"), x="bin_start", kind="kde"),
        without=_MissingColumn(
            build=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="kde"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_displot_ecdf_x": _RemedyRun(
        refused=lambda: fm.displot(_XY_ROWS, x=X("v", bin=True), kind="ecdf"),
        names=_names_bin("v"),
        remedied=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="ecdf").transform(
            _per_row_bin("v")
        ),
        sibling=lambda: fm.displot(_prebinned(_XY_ROWS, "v"), x="bin_start", kind="ecdf"),
        without=_MissingColumn(
            build=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="ecdf"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "per_row_bin_mark_smooth_y": _RemedyRun(
        refused=lambda: fm.Chart(_XY_ROWS).mark_smooth().encode(x="v", y=Y("w", bin=True)).to_svg(),
        names=_names_bin("w"),
        remedied=lambda: (
            fm.Chart(_XY_ROWS)
            .transform(_per_row_bin("w"))
            .mark_smooth()
            .encode(x="v", y="bin_start")
        ),
        sibling=lambda: (
            fm.Chart(_prebinned(_XY_ROWS, "w")).mark_smooth().encode(x="v", y="bin_start")
        ),
        without=_MissingColumn(
            build=lambda: fm.Chart(_XY_ROWS).mark_smooth().encode(x="v", y="bin_start"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    # residplot's remedy carries dropna=False: its default drops nulls over
    # x/y in the input frame, before the remedy's column exists.
    "per_row_bin_residplot_x": _RemedyRun(
        refused=lambda: fm.residplot(_XY_ROWS, x=X("v", bin=True), y="w"),
        names=_names_bin("v", " with dropna=False"),
        remedied=lambda: fm.residplot(_XY_ROWS, x="bin_start", y="w", dropna=False).transform(
            _per_row_bin("v")
        ),
        sibling=lambda: fm.residplot(_prebinned(_XY_ROWS, "v"), x="bin_start", y="w", dropna=False),
        without=_MissingColumn(
            build=lambda: fm.residplot(_XY_ROWS, x="bin_start", y="w").transform(_per_row_bin("v")),
            error=_COLUMN_NOT_FOUND,
            column="bin_start",
        ),
    ),
    "per_row_bin_residplot_y": _RemedyRun(
        refused=lambda: fm.residplot(_XY_ROWS, x="v", y=Y("w", bin=True)),
        names=_names_bin("w", " with dropna=False"),
        remedied=lambda: fm.residplot(_XY_ROWS, x="v", y="bin_start", dropna=False).transform(
            _per_row_bin("w")
        ),
        sibling=lambda: fm.residplot(_prebinned(_XY_ROWS, "w"), x="v", y="bin_start", dropna=False),
        without=_MissingColumn(
            build=lambda: fm.residplot(_XY_ROWS, x="v", y="bin_start").transform(_per_row_bin("w")),
            error=_COLUMN_NOT_FOUND,
            column="bin_start",
        ),
    ),
    # transform_aggregate: aggregate ahead of the statistic, bind the output.
    "transform_aggregate_mark_histogram_binned_x": _RemedyRun(
        refused=lambda: (
            fm.Chart(_GROUPED_XY_ROWS).mark_histogram().encode(x=X("v", aggregate="mean")).to_svg()
        ),
        names=_names_aggregate(),
        remedied=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .transform(_per_group_mean())
            .mark_histogram()
            .encode(x="mean_v")
        ),
        sibling=lambda: fm.Chart(_PREAGGREGATED_ROWS).mark_histogram().encode(x="mean_v"),
        without=_MissingColumn(
            build=lambda: fm.Chart(_GROUPED_XY_ROWS).mark_histogram().encode(x="mean_v"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_mark_density_x": _RemedyRun(
        refused=lambda: (
            fm.Chart(_GROUPED_XY_ROWS).mark_density().encode(x=X("v", aggregate="mean")).to_svg()
        ),
        names=_names_aggregate(),
        remedied=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .transform(_per_group_mean())
            .mark_density()
            .encode(x="mean_v")
        ),
        sibling=lambda: fm.Chart(_PREAGGREGATED_ROWS).mark_density().encode(x="mean_v"),
        without=_MissingColumn(
            build=lambda: fm.Chart(_GROUPED_XY_ROWS).mark_density().encode(x="mean_v"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_displot_kde_x": _RemedyRun(
        refused=lambda: fm.displot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), kind="kde"),
        names=_names_aggregate(),
        remedied=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="kde").transform(
            _per_group_mean()
        ),
        sibling=lambda: fm.displot(_PREAGGREGATED_ROWS, x="mean_v", kind="kde"),
        without=_MissingColumn(
            build=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="kde"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    # The ECDF counts across every row of its partition, so the groups the
    # aggregate wrote do not split its line: one step line, two under a hue.
    "transform_aggregate_displot_ecdf_x": _RemedyRun(
        refused=lambda: fm.displot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), kind="ecdf"),
        names=_names_aggregate(),
        remedied=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="ecdf").transform(
            _per_group_mean()
        ),
        sibling=lambda: fm.displot(_PREAGGREGATED_ROWS, x="mean_v", kind="ecdf"),
        without=_MissingColumn(
            build=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="ecdf"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_displot_ecdf_x_hue": _RemedyRun(
        refused=lambda: fm.displot(
            _GROUPED_HUE_ROWS, x=X("v", aggregate="mean"), kind="ecdf", hue="h"
        ),
        names=_names_aggregate(),
        remedied=lambda: fm.displot(_GROUPED_HUE_ROWS, x="mean_v", kind="ecdf", hue="h").transform(
            _per_group_and_hue_mean()
        ),
        sibling=lambda: fm.displot(_PREAGGREGATED_HUE_ROWS, x="mean_v", kind="ecdf", hue="h"),
        without=_MissingColumn(
            build=lambda: fm.displot(_GROUPED_HUE_ROWS, x="mean_v", kind="ecdf", hue="h"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_mark_smooth_x": _RemedyRun(
        refused=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .mark_smooth(method="lm")
            .encode(x=X("v", aggregate="mean"), y="w")
            .to_svg()
        ),
        names=_names_aggregate(),
        remedied=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .transform(_per_group_mean())
            .mark_smooth(method="lm")
            .encode(x="mean_v", y="mean_w")
        ),
        sibling=lambda: (
            fm.Chart(_PREAGGREGATED_ROWS).mark_smooth(method="lm").encode(x="mean_v", y="mean_w")
        ),
        without=_MissingColumn(
            build=lambda: (
                fm.Chart(_GROUPED_XY_ROWS).mark_smooth(method="lm").encode(x="mean_v", y="mean_w")
            ),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_mark_smooth_y": _RemedyRun(
        refused=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .mark_smooth(method="lm")
            .encode(x="v", y=Y("w", aggregate="mean"))
            .to_svg()
        ),
        names=_names_aggregate(),
        remedied=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .transform(_per_group_mean())
            .mark_smooth(method="lm")
            .encode(x="mean_v", y="mean_w")
        ),
        sibling=lambda: (
            fm.Chart(_PREAGGREGATED_ROWS).mark_smooth(method="lm").encode(x="mean_v", y="mean_w")
        ),
        without=_MissingColumn(
            build=lambda: (
                fm.Chart(_GROUPED_XY_ROWS).mark_smooth(method="lm").encode(x="mean_v", y="mean_w")
            ),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_lmplot_lm_x": _RemedyRun(
        refused=lambda: fm.lmplot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), y="w", method="lm"),
        names=_names_aggregate(),
        remedied=lambda: fm.lmplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="lm").transform(
            _per_group_mean()
        ),
        sibling=lambda: fm.lmplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", method="lm"),
        without=_MissingColumn(
            build=lambda: fm.lmplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="lm"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_regplot_robust_y": _RemedyRun(
        refused=lambda: fm.regplot(
            _GROUPED_XY_ROWS, x="v", y=Y("w", aggregate="mean"), method="robust"
        ),
        names=_names_aggregate(),
        remedied=lambda: fm.regplot(
            _GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="robust"
        ).transform(_per_group_mean()),
        sibling=lambda: fm.regplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", method="robust"),
        without=_MissingColumn(
            build=lambda: fm.regplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="robust"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_residplot_x": _RemedyRun(
        refused=lambda: fm.residplot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), y="w"),
        names=_names_aggregate(" with dropna=False"),
        remedied=lambda: fm.residplot(
            _GROUPED_XY_ROWS, x="mean_v", y="mean_w", dropna=False
        ).transform(_per_group_mean()),
        sibling=lambda: fm.residplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", dropna=False),
        without=_MissingColumn(
            build=lambda: fm.residplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w").transform(
                _per_group_mean()
            ),
            error=_COLUMN_NOT_FOUND,
            column="mean_v",
        ),
    ),
    "transform_aggregate_residplot_y": _RemedyRun(
        refused=lambda: fm.residplot(_GROUPED_XY_ROWS, x="v", y=Y("w", aggregate="mean")),
        names=_names_aggregate(" with dropna=False"),
        remedied=lambda: fm.residplot(
            _GROUPED_XY_ROWS, x="mean_v", y="mean_w", dropna=False
        ).transform(_per_group_mean()),
        sibling=lambda: fm.residplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", dropna=False),
        without=_MissingColumn(
            build=lambda: fm.residplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w").transform(
                _per_group_mean()
            ),
            error=_COLUMN_NOT_FOUND,
            column="mean_v",
        ),
    ),
    # mark_contour's x/y take the same generic per-row Bin/transform_aggregate
    # remedies as mark_density's bivariate path (the same kernel-density
    # reason, the same two generic remedies); mark_hex's own aggregate= is
    # the hex-grid-specific bin_size= remedy below, but its x/y aggregate=
    # (pre-aggregating rows before the hexagons bin them) takes the same
    # generic transform_aggregate remedy too.
    "per_row_bin_mark_contour_x": _RemedyRun(
        refused=lambda: (
            fm.Chart(_XY_ROWS).mark_contour().encode(x=X("v", bin=True), y="w").to_svg()
        ),
        names=_names_bin("v"),
        remedied=lambda: (
            fm.Chart(_XY_ROWS)
            .transform(_per_row_bin("v"))
            .mark_contour()
            .encode(x="bin_start", y="w")
        ),
        sibling=lambda: (
            fm.Chart(_prebinned(_XY_ROWS, "v")).mark_contour().encode(x="bin_start", y="w")
        ),
        without=_MissingColumn(
            build=lambda: fm.Chart(_XY_ROWS).mark_contour().encode(x="bin_start", y="w"),
            error=ValueError,
            column="bin_start",
        ),
    ),
    "transform_aggregate_mark_contour_x": _RemedyRun(
        refused=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .mark_contour()
            .encode(x=X("v", aggregate="mean"), y="w")
            .to_svg()
        ),
        names=_names_aggregate(),
        remedied=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .transform(_per_group_mean())
            .mark_contour()
            .encode(x="mean_v", y="mean_w")
        ),
        sibling=lambda: fm.Chart(_PREAGGREGATED_ROWS).mark_contour().encode(x="mean_v", y="mean_w"),
        without=_MissingColumn(
            build=lambda: fm.Chart(_GROUPED_XY_ROWS).mark_contour().encode(x="mean_v", y="mean_w"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    "transform_aggregate_mark_hex_x": _RemedyRun(
        refused=lambda: (
            fm.Chart(_GROUPED_XY_ROWS).mark_hex().encode(x=X("v", aggregate="mean"), y="w").to_svg()
        ),
        names=_names_aggregate(),
        remedied=lambda: (
            fm.Chart(_GROUPED_XY_ROWS)
            .transform(_per_group_mean())
            .mark_hex()
            .encode(x="mean_v", y="mean_w")
        ),
        sibling=lambda: fm.Chart(_PREAGGREGATED_ROWS).mark_hex().encode(x="mean_v", y="mean_w"),
        without=_MissingColumn(
            build=lambda: fm.Chart(_GROUPED_XY_ROWS).mark_hex().encode(x="mean_v", y="mean_w"),
            error=ValueError,
            column="mean_v",
        ),
    ),
    # Named-keyword remedies: the door's own spelling, run as written,
    # renders the same chart a different route already draws. displot's
    # bin rule and density flag are reached post-hoc, after displot's own
    # call-time pre-check has already returned; jointplot's density flag
    # has no pre-check at all, since jointplot builds its marginal
    # histogram directly.
    "bins_displot_posthoc_x": _RemedyRun(
        refused=lambda: (
            fm.displot(_HIST_ROWS, x="v", kind="hist")
            .encode(x=X("v", bin={"bin_count": 3}))
            .to_svg()
        ),
        names="State the rule with displot's bins=, or drop bin=.",
        remedied=lambda: fm.displot(_HIST_ROWS, x="v", kind="hist", bins=3),
        sibling=lambda: fm.Chart(_HIST_ROWS).mark_histogram(bins=3).encode(x="v"),
        without=_DifferentChart(lambda: fm.displot(_HIST_ROWS, x="v", kind="hist")),
    ),
    "bins_jointplot_marginal_kws_x": _RemedyRun(
        refused=lambda: fm.jointplot(_XY_ROWS, x=X("v", bin={"bin_count": 3}), y="w").to_svg(),
        names="State the rule with marginal_kws={'bins': ...}, or drop bin=.",
        remedied=lambda: fm.jointplot(_XY_ROWS, x="v", y="w", marginal_kws={"bins": 3}),
        sibling=lambda: fm.jointplot(_XY_ROWS, x="v", y="w", marginal_kws={"bin_count": 3}),
        without=_DifferentChart(lambda: fm.jointplot(_XY_ROWS, x="v", y="w")),
    ),
    # mark_hex's own grid knob: run through jointplot(kind="hex")'s
    # joint_kws=, the one other door that reaches mark_hex's bin_size=, so
    # the sibling crosses a real door boundary rather than repeating the
    # same call.
    "bin_size_mark_hex_x": _RemedyRun(
        refused=lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x=X("v", bin=True), y="w").to_svg(),
        names=_HEX_BIN_REMEDY,
        remedied=lambda: fm.Chart(_XY_ROWS).mark_hex(bin_size=5.0).encode(x="v", y="w"),
        sibling=lambda: (
            fm.jointplot(_XY_ROWS, x="v", y="w", kind="hex", joint_kws={"bin_size": 5.0}).center
        ),
        without=_DifferentChart(lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x="v", y="w")),
    ),
    # mark_raster's own grid knob: no other door reaches it, so the sibling
    # is resolution='s own alternate spelling (an explicit (nx, ny) tuple
    # equal to the plain int), proving the remedy's keyword is typeable
    # more than one way, not just repeating the same call.
    "resolution_mark_raster_x": _RemedyRun(
        refused=lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x=X("v", bin=True), y="w").to_svg(),
        names=_RASTER_BIN_REMEDY,
        remedied=lambda: fm.Chart(_XY_ROWS).mark_raster(resolution=20).encode(x="v", y="w"),
        sibling=lambda: fm.Chart(_XY_ROWS).mark_raster(resolution=(20, 20)).encode(x="v", y="w"),
        without=_DifferentChart(lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x="v", y="w")),
    ),
    # jointplot's own spelling of mark_hex's grid knob, run on jointplot
    # itself rather than crossing back to the mark door (the mirror of
    # bin_size_mark_hex_x above).
    "bin_size_jointplot_hex_x": _RemedyRun(
        refused=lambda: fm.jointplot(_XY_ROWS, x=X("v", bin=True), y="w", kind="hex").to_svg(),
        names="Size the cells with joint_kws={'bin_size': ...}, or drop bin=.",
        remedied=lambda: fm.jointplot(
            _XY_ROWS, x="v", y="w", kind="hex", joint_kws={"bin_size": 5.0}
        ),
        sibling=lambda: fm.jointplot(_XY_ROWS, x="v", y="w", kind="hex", joint_kws={"bin_size": 5}),
        without=_DifferentChart(lambda: fm.jointplot(_XY_ROWS, x="v", y="w", kind="hex")),
    ),
}


@pytest.mark.parametrize("run", list(_REMEDY_RUNS.values()), ids=list(_REMEDY_RUNS))
def test_a_named_remedy_on_its_own_door_matches_the_door_over_a_precomputed_frame(
    run: _RemedyRun,
) -> None:
    """The remedy the refusal names, run on that door, draws byte-identically to
    the same door drawing the same thing a different way; its own negative
    control proves the remedy's step actually mattered -- a generic
    remedy's column is never computed without it, a named-keyword remedy's
    kwarg left out renders a different chart.
    """
    with pytest.raises(ValueError) as excinfo:
        run.refused()
    assert run.names in str(excinfo.value)
    remedied_svg = run.remedied().to_svg()
    assert remedied_svg == run.sibling().to_svg()
    if isinstance(run.without, _MissingColumn):
        with pytest.raises(run.without.error, match=rf"{run.without.column}['\"]? not found"):
            run.without.build().to_svg()
    else:
        assert run.without.build().to_svg() != remedied_svg


@pytest.mark.parametrize(
    "run_id,steps",
    [("transform_aggregate_displot_ecdf_x", 1), ("transform_aggregate_displot_ecdf_x_hue", 2)],
)
def test_the_ecdf_aggregate_remedy_draws_one_step_line_per_hue_level(run_id: str, steps: int):
    """Two blank renders are byte-equal too, so the remedied ECDF's step lines
    are counted: one without a hue, one per level with one.
    """
    svg = _REMEDY_RUNS[run_id].remedied().to_svg()
    assert len(step_after_paths(svg)) == steps


_X_BINS_REMEDY_TEXT = "Bin the predictor with x_bins=<number of bins>, the fit's own binning"
#: A nonlinear response, so a fit over 4 bin means differs from one over every row.
_SQUARE_ROWS = pl.DataFrame(
    {"v": [float(i) for i in range(1, 21)], "w": [float(i) ** 2 for i in range(1, 21)]}
)


def test_remedy_x_bins_on_mark_smooth_matches_the_same_pre_aggregation():
    """``mark_smooth(x_bins=4)`` over a nonlinear response matches the same fit over
    rows pre-aggregated in polars into the same 4 equal-width bins (mean x, mean y);
    without ``x_bins=`` the fit reads every row and differs.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_SQUARE_ROWS).mark_smooth(method="lm").encode(x=X("v", bin=True), y="w").to_svg()
    assert _X_BINS_REMEDY_TEXT in str(excinfo.value)

    remedied = (
        fm.Chart(_SQUARE_ROWS).mark_smooth(method="lm", x_bins=4).encode(x="v", y="w").to_svg()
    )
    sibling = (
        fm.Chart(_equal_width_binned_mean(_SQUARE_ROWS, bins=4))
        .mark_smooth(method="lm")
        .encode(x="v", y="w")
        .to_svg()
    )
    assert remedied == sibling
    without = fm.Chart(_SQUARE_ROWS).mark_smooth(method="lm").encode(x="v", y="w").to_svg()
    assert without != sibling


@pytest.mark.parametrize("door", ["regplot", "lmplot"])
def test_remedy_x_bins_on_a_fit_figure_matches_the_same_pre_aggregation(door: str):
    """``regplot``/``lmplot(x_bins=4, method='lm')`` fit the same polyline as the
    same method over the frame pre-aggregated into the same 4 bins, with both
    calls' x scale domain pinned so they resolve over one shared range; without
    ``x_bins=`` the polyline differs.
    """
    call = {"regplot": fm.regplot, "lmplot": fm.lmplot}[door]
    with pytest.raises(ValueError) as excinfo:
        call(_SQUARE_ROWS, x=X("v", bin=True), y="w", method="lm")
    assert _X_BINS_REMEDY_TEXT in str(excinfo.value)

    x = X("v", scale=fm.LinearScale(domain=(1.0, 20.0)))
    binned = _equal_width_binned_mean(_SQUARE_ROWS, bins=4)
    remedied = call(_SQUARE_ROWS, x=x, y="w", method="lm", x_bins=4, scatter=False, ci=None)
    sibling = call(binned, x=x, y="w", method="lm", scatter=False, ci=None)
    without = call(_SQUARE_ROWS, x=x, y="w", method="lm", scatter=False, ci=None)
    assert polylines(remedied.to_svg()) == polylines(sibling.to_svg())
    assert polylines(without.to_svg()) != polylines(sibling.to_svg())


def test_remedy_bins_parameters_draw_the_rule_the_channel_stated():
    """The rule a refused ``bin={'bin_count': 3}`` stated, moved to the door's own
    ``bins=``/``bin_count=``, draws 3 bars; ``displot(bins=3)`` is byte-equal to
    ``mark_histogram(bins=3)``; ``bin_width=5.0`` draws 4 bars over 1..20; and
    without the rule the default draws a different count.
    """
    rule = {"bin_count": 3}
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_HIST_ROWS).mark_histogram().encode(x=X("v", bin=rule)).to_svg()
    assert "State the rule with mark_histogram's bins=, bin_count= or bin_width=" in str(
        excinfo.value
    )
    with pytest.raises(ValueError) as excinfo:
        fm.displot(_HIST_ROWS, x=X("v", bin=rule), kind="hist")
    assert "State the rule with displot's bins=" in str(excinfo.value)

    mark_svg = fm.Chart(_HIST_ROWS).mark_histogram(bins=3).encode(x="v").to_svg()
    assert fm.displot(_HIST_ROWS, x="v", kind="hist", bins=3).to_svg() == mark_svg
    assert len(data_bar_rects(mark_svg)) == 3
    bin_count_svg = fm.Chart(_HIST_ROWS).mark_histogram(bin_count=3).encode(x="v").to_svg()
    assert len(data_bar_rects(bin_count_svg)) == 3
    bin_width_svg = fm.Chart(_HIST_ROWS).mark_histogram(bin_width=5.0).encode(x="v").to_svg()
    assert len(data_bar_rects(bin_width_svg)) == 4
    without = fm.Chart(_HIST_ROWS).mark_histogram().encode(x="v").to_svg()
    assert len(data_bar_rects(without)) != 3


@dataclass(frozen=True)
class _CountRemedy:
    """A refused fieldless count, and the door setting its remedy names.

    *door* builds the chart from its keyword arguments and the value
    channel; *refused_kwargs* draws a density or a share, *remedy_kwargs*
    the count the message names.
    """

    door: Callable[..., fm.Chart]
    refused_kwargs: dict
    remedy_kwargs: dict
    names: str
    count: Y = Y(aggregate="count")
    plain: Y | None = None


def _histogram_door(**kwargs: Any) -> fm.Chart:
    y = kwargs.pop("y", None)
    encode = {"x": "v", "color": "g"} if "groupby" in kwargs else {"x": "v"}
    if y is not None:
        encode["y"] = y
    rows = _GROUPED_HIST_ROWS if "groupby" in kwargs else _HIST_ROWS
    return fm.Chart(rows).mark_histogram(**kwargs).encode(**encode)


def _displot_door(**kwargs: Any) -> fm.Chart:
    rows = _GROUPED_HIST_ROWS if "hue" in kwargs else _HIST_ROWS
    return fm.displot(rows, x="v", kind="hist", **kwargs)


def _displot_posthoc_door(**kwargs: Any) -> fm.Chart:
    """*kwargs*' own ``y=`` moved to a ``.encode()`` written after the call returns."""
    y = kwargs.pop("y", None)
    rows = _GROUPED_HIST_ROWS if "hue" in kwargs else _HIST_ROWS
    chart = fm.displot(rows, x="v", kind="hist", **kwargs)
    return chart.encode(y=y) if y is not None else chart


def _pairplot_door(**kwargs: Any) -> fm.Chart:
    y = kwargs.pop("y")
    return fm.pairplot(
        _XY_HUE_ROWS, vars=["v", "w"], hue="g", kind="hist", diag_kind=None, encode={"y": y}
    )


_COUNT_REMEDIES: dict[str, _CountRemedy] = {
    "mark_histogram_density_false": _CountRemedy(
        door=_histogram_door,
        refused_kwargs={"density": True},
        remedy_kwargs={"density": False},
        names="draw the count with density=False.",
    ),
    "displot_stat_count": _CountRemedy(
        door=_displot_door,
        refused_kwargs={"stat": "density"},
        remedy_kwargs={"stat": "count"},
        names="draw the count with stat='count'.",
    ),
    "mark_histogram_multiple_stack": _CountRemedy(
        door=_histogram_door,
        refused_kwargs={"multiple": "fill", "groupby": ["g"]},
        remedy_kwargs={"multiple": "stack", "groupby": ["g"]},
        names="draw the count with multiple='stack'.",
    ),
    "displot_multiple_stack": _CountRemedy(
        door=_displot_door,
        refused_kwargs={"multiple": "fill", "hue": "g"},
        remedy_kwargs={"multiple": "stack", "hue": "g"},
        names="draw the count with multiple='stack'.",
    ),
    "mark_histogram_position_stack": _CountRemedy(
        door=_histogram_door,
        refused_kwargs={"groupby": ["g"], "position": fm.Stack(offset="normalize")},
        remedy_kwargs={"groupby": ["g"], "position": fm.Stack()},
        names="draw the count with position=fm.Stack().",
    ),
    "mark_histogram_stack_true": _CountRemedy(
        door=_histogram_door,
        refused_kwargs={"groupby": ["g"], "y": Y(aggregate="count", stack="normalize")},
        remedy_kwargs={"groupby": ["g"]},
        names="draw the count with stack=True.",
        count=Y(aggregate="count", stack=True),
        plain=Y(stack=True),
    ),
    "displot_posthoc_stat_count": _CountRemedy(
        door=_displot_posthoc_door,
        refused_kwargs={"stat": "density"},
        remedy_kwargs={"stat": "count"},
        names="draw the count with stat='count'.",
    ),
    "displot_posthoc_multiple_stack": _CountRemedy(
        door=_displot_posthoc_door,
        refused_kwargs={"multiple": "fill", "hue": "g"},
        remedy_kwargs={"multiple": "stack", "hue": "g"},
        names="draw the count with multiple='stack'.",
    ),
    "pairplot_stack_true": _CountRemedy(
        door=_pairplot_door,
        refused_kwargs={"y": Y(aggregate="count", stack="normalize")},
        remedy_kwargs={},
        names="draw the count with stack=True.",
        count=Y(aggregate="count", stack=True),
        plain=Y(stack=True),
    ),
}


@pytest.mark.parametrize("remedy", list(_COUNT_REMEDIES.values()), ids=list(_COUNT_REMEDIES))
def test_a_count_remedy_draws_the_same_chart_as_no_value_channel(remedy: _CountRemedy):
    """The setting a refused fieldless count names draws, with the explicit count,
    byte-identically to the same chart without it; the refused setting raises.
    """
    refused_kwargs = {"y": remedy.count, **remedy.refused_kwargs}
    with pytest.raises(ValueError) as excinfo:
        remedy.door(**refused_kwargs).to_svg()
    assert str(excinfo.value).endswith(remedy.names)

    plain_kwargs = dict(remedy.remedy_kwargs)
    if remedy.plain is not None:
        plain_kwargs["y"] = remedy.plain
    remedied = remedy.door(**remedy.remedy_kwargs, y=remedy.count).to_svg()
    assert remedied == remedy.door(**plain_kwargs).to_svg()


def _largest_numeric_tick(svg: str, axis: str) -> float:
    labels = axis_tick_labels(svg, axis=axis)
    numeric = [
        float(label) for label in labels if label.replace(".", "", 1).replace("-", "", 1).isdigit()
    ]
    assert numeric, labels
    return max(numeric)


@pytest.mark.parametrize("kwarg", ["aggregate", "bin"])
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_a_histogram_value_axis_remedy_runs_as_its_message_writes_it(orientation: str, kwarg: str):
    """The bar-mark per-bin aggregate (``aggregate=``) and the ``mark_hex`` two-field
    count (``bin=``) are evaluated from the message's own text over the same
    rows. Each draws ``w`` on the value axis, whose ticks pass the row count no
    bin count can reach; the bare-field histogram the remedy steers away from
    draws counts there instead.
    """
    bin_axis, value_axis = ("y", "x") if orientation == "horizontal" else ("x", "y")
    va, ba = value_axis.upper(), bin_axis.upper()
    option = {"aggregate": "mean"} if kwarg == "aggregate" else {"bin": True}
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_XY_ROWS).mark_histogram(orientation=orientation).encode(
            **{bin_axis: "v", value_axis: _axis_channel(value_axis, "w", **option)}
        ).to_svg()
    if kwarg == "aggregate":
        call = (
            f"fm.Chart(data).mark_bar().encode({bin_axis}=fm.{ba}('v', bin=True), "
            f"{value_axis}=fm.{va}('w', aggregate='mean'))"
        )
    else:
        call = f"fm.Chart(data).mark_hex().encode({bin_axis}='v', {value_axis}='w')"
    assert f"{call}, or drop {value_axis}=" in str(excinfo.value)

    remedied = eval(call, {"fm": fm, "data": _XY_ROWS}).to_svg()
    assert _largest_numeric_tick(remedied, value_axis) > len(_XY_ROWS)
    bare_field = (
        fm.Chart(_XY_ROWS)
        .mark_histogram(orientation=orientation)
        .encode(**{bin_axis: "v", value_axis: "w"})
        .to_svg()
    )
    assert _largest_numeric_tick(bare_field, value_axis) <= len(_XY_ROWS)


@pytest.mark.parametrize("kwarg", ["aggregate", "bin"])
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_a_histogram_value_axis_remedy_with_no_binned_field_binds_one(orientation: str, kwarg: str):
    """With nothing bound on the binned axis, the remedy binds the field to bin
    there and drops the value-axis field, styling that axis instead: the bars
    are the plain histogram's and the axis carries the title. Dropping the
    value-axis field alone leaves no field to bin.
    """
    bin_axis, value_axis = ("y", "x") if orientation == "horizontal" else ("x", "y")
    va = value_axis.upper()
    option = {"aggregate": "mean"} if kwarg == "aggregate" else {"bin": True}
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_XY_ROWS).mark_histogram(orientation=orientation).encode(
            **{value_axis: _axis_channel(value_axis, "w", **option)}
        ).to_svg()
    assert str(excinfo.value).endswith(
        f"Bind the field to bin on {bin_axis}= and drop {value_axis}= "
        f"(style the axis with fm.{va}(title=...) and no field)."
    )

    styled = _axis_channel(value_axis, None, title="rows per bin")
    remedied = (
        fm.Chart(_XY_ROWS)
        .mark_histogram(orientation=orientation)
        .encode(**{bin_axis: "v", value_axis: styled})
        .to_svg()
    )
    plain = (
        fm.Chart(_XY_ROWS)
        .mark_histogram(orientation=orientation)
        .encode(**{bin_axis: "v"})
        .to_svg()
    )
    assert data_bar_rects(remedied) == data_bar_rects(plain)
    title = x_axis_title(remedied) if value_axis == "x" else y_axis_title(remedied)
    assert title == "rows per bin"
    with pytest.raises(ValueError, match=rf"requires \.encode\({bin_axis}=\.\.\.\)"):
        fm.Chart(_XY_ROWS).mark_histogram(orientation=orientation).encode(
            **{value_axis: styled}
        ).to_svg()


def test_remedy_mark_function_chained_is_byte_equal_to_the_plus_route():
    """The chained ``mark_point().encode(...).mark_function(fn)`` form the refusal
    names is byte-equal to the same point layer composed with a bare
    ``mark_function`` chart via ``+``; with no data layer before it, the same
    binding is refused.
    """
    fn = lambda xs: xs  # noqa: E731
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_HIST_ROWS).encode(x=X("v", bin=True)).mark_function(
            fn, domain=[0.0, 20.0]
        ).to_svg()
    assert "(chart.mark_point().encode(...).mark_function(fn)), which honors bin=" in str(
        excinfo.value
    )
    chained = (
        fm.Chart(_HIST_ROWS)
        .mark_point()
        .encode(x=X("v", bin=True), y="v")
        .mark_function(fn, domain=[0.0, 20.0])
        .to_svg()
    )
    plus = (
        fm.Chart(_HIST_ROWS).mark_point().encode(x=X("v", bin=True), y="v")
        + fm.Chart(None).mark_function(fn, domain=[0.0, 20.0])
    ).to_svg()
    assert chained == plus


# ---------------------------------------------------------------------------
# Honored rows: byte-identical controls, by sibling route or by a committed
# digest of the same chart (this file never reads a scratch directory at
# test time).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiple", ["layer", "stack", "dodge"])
def test_a_histogram_honors_a_fieldless_count_under_every_non_fill_multiple(multiple: str):
    rows = _GROUPED_HIST_ROWS
    groupby = {"groupby": ["g"]} if multiple != "layer" else {}
    plain = (
        fm.Chart(rows)
        .mark_histogram(multiple=multiple, **groupby)
        .encode(x="v", color="g")
        .to_svg()
    )
    explicit = (
        fm.Chart(rows)
        .mark_histogram(multiple=multiple, **groupby)
        .encode(x="v", color="g", y=Y(aggregate="count"))
        .to_svg()
    )
    assert plain == explicit


def test_a_histogram_honors_a_fieldless_count_under_cumulative():
    plain = fm.Chart(_HIST_ROWS).mark_histogram(cumulative=True).encode(x="v").to_svg()
    explicit = (
        fm.Chart(_HIST_ROWS)
        .mark_histogram(cumulative=True)
        .encode(x="v", y=Y(aggregate="count"))
        .to_svg()
    )
    assert plain == explicit


def test_a_histogram_honors_a_fieldless_count_under_multiple_dodge_with_a_normalize_position():
    """``multiple='dodge'`` replaces a caller's own ``position=``, so a normalize
    position set alongside it never reaches the axis -- the count stays honored.
    """
    rows = _GROUPED_HIST_ROWS
    plain = (
        fm.Chart(rows)
        .mark_histogram(multiple="dodge", groupby=["g"], position=fm.Stack(offset="normalize"))
        .encode(x="v", color="g")
        .to_svg()
    )
    explicit = (
        fm.Chart(rows)
        .mark_histogram(multiple="dodge", groupby=["g"], position=fm.Stack(offset="normalize"))
        .encode(x="v", color="g", y=Y(aggregate="count"))
        .to_svg()
    )
    assert plain == explicit


def test_a_histogram_honors_bin_true_on_its_binned_axis():
    """``mark_histogram``'s own axis bins anyway, so a redundant ``bin=True`` renders unchanged."""
    plain = fm.Chart(_HIST_ROWS).mark_histogram().encode(x="v").to_svg()
    redundant = fm.Chart(_HIST_ROWS).mark_histogram().encode(x=X("v", bin=True)).to_svg()
    assert plain == redundant


def test_displot_hist_honors_bin_true_and_a_fieldless_count():
    plain = fm.displot(_HIST_ROWS, x="v", kind="hist").to_svg()
    bin_true = fm.displot(_HIST_ROWS, x=X("v", bin=True), kind="hist").to_svg()
    assert plain == bin_true
    explicit_count = fm.displot(_HIST_ROWS, x="v", y=Y(aggregate="count"), kind="hist").to_svg()
    assert plain == explicit_count


def test_encode_x_none_leaves_the_keyword_binding_in_place():
    plain = fm.displot(_HIST_ROWS, x="v", kind="hist").to_svg()
    with_none = fm.displot(_HIST_ROWS, x="v", kind="hist", encode={"x": None}).to_svg()
    assert plain == with_none


def test_mark_function_over_a_parent_data_layer_is_unaffected_by_its_bin():
    """A parent layer's own ``bin=`` is honored by its own lowering; ``mark_function``'s
    grid is unaffected either way. The render is pinned by digest.
    """
    chart = (
        fm.Chart(_HIST_ROWS)
        .mark_point()
        .encode(x=X("v", bin=True), y="v")
        .mark_function(lambda xs: xs, domain=[0.0, 20.0])
    )
    svg = chart.to_svg()
    assert _digest(svg) == "303f62297cae8cd300203206bb872c97760aa50ad471ac7df1db1a5efcb3d79f"


def test_displot_rug_is_unaffected_by_a_channel_bin_or_aggregate():
    """``kind="rug"`` draws every row through the plain lowering, which consumes the option."""
    bin_svg = fm.displot(_HIST_ROWS, x=X("v", bin=True), kind="rug").to_svg()
    assert _digest(bin_svg) == "35e78d7e198e6af2049519c4714753ffafe831702da2487371bc28b1331aa551"
    agg_svg = fm.displot(_HIST_ROWS, x=X("v", aggregate="mean"), kind="rug").to_svg()
    assert _digest(agg_svg) == "56c365e689ff98f2238169a118507bf85d07fb89cace127da3f2605ceb9b211b"


def test_a_primitive_marks_own_bin_and_aggregate_lowering_is_unaffected():
    """The three ``_encode_lowering`` sites that consume ``bin=``/``aggregate=`` directly
    on a primitive mark (not through a composite mark's desugar) keep rendering, including
    the folded-aggregate site (a bin on x plus a non-count aggregate on y).
    """
    bin_count = (
        fm.Chart(_HIST_ROWS).mark_bar().encode(x=X("v", bin=True), y=Y(aggregate="count")).to_svg()
    )
    assert _digest(bin_count) == "73e0b54f410e2bae8124aa5a7d7a51203792e355bbc1bd1a49d787a92059b2d5"
    bin_folded_aggregate = (
        fm.Chart(_HIST_ROWS)
        .mark_bar()
        .encode(x=X("v", bin=True), y=Y("v", aggregate="mean"))
        .to_svg()
    )
    assert (
        _digest(bin_folded_aggregate)
        == "f880e7138331feb43c35a2a849228fb38dae688749ae8c731f35d553e43ca792"
    )
    bin_less_aggregate = (
        fm.Chart(_HIST_ROWS).mark_point().encode(x="v", y=Y("v", aggregate="mean")).to_svg()
    )
    assert (
        _digest(bin_less_aggregate)
        == "618b6f360bda0f5fd5b3ed5de5c712ee0da53db6572caeec4fea6b033f7660f6"
    )


# ---------------------------------------------------------------------------
# mark_hex's color-aggregate fold: a matching aggregate= (or an unset one) is
# consumed; a different op is refused. mark_raster and mark_contour never
# fold a color aggregate=, always refused.
# ---------------------------------------------------------------------------


def _hex_mean_target() -> str:
    """``mark_hex``'s own ``aggregate='mean', field='z'`` call -- the render every
    "consumed as mean" row below must match.
    """
    return (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(bin_size=4.0, aggregate="mean", field="z")
        .encode(x="v", y="w")
        .to_svg()
    )


def _hex_count_target() -> str:
    """``mark_hex``'s own plain (unset, count) call -- the render every
    "consumed as count" row below must match.
    """
    return fm.Chart(_XYZ_ROWS).mark_hex(bin_size=4.0).encode(x="v", y="w").to_svg()


_HEX_MISMATCH_TEXT = (
    "mark_hex(color=Color('z', aggregate='mean')) is not supported: the cells are "
    "colored by the statistic's own value, so aggregate= would be ignored. Aggregate "
    "the cells with mark_hex(aggregate='mean', field='z'), or drop aggregate=."
)


@pytest.mark.parametrize(
    "build,expected",
    [
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="count")
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            _HEX_MISMATCH_TEXT,
            id="explicit_count_differs",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0)
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            _hex_mean_target,
            id="unset_consumed",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="mean")
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            _hex_mean_target,
            id="equal_mean_consumed",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="sum", field="z")
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            _HEX_MISMATCH_TEXT,
            id="different_op_refused",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="mean", field="z")
                .encode(x="v", y="w", color=fm.Color(aggregate="sum"))
                .to_svg()
            ),
            "mark_hex(color=Color(aggregate='sum')) is not supported: the cells are "
            "colored by the statistic's own value, so aggregate= would be ignored. Drop "
            "aggregate=.",
            id="fieldless_sum_refused",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="mean", field="z")
                .encode(x="v", y="w", color=fm.Color(aggregate="count"))
                .to_svg()
            ),
            "mark_hex(color=Color(aggregate='count')) is not supported: the cells are "
            "colored by the statistic's own value, so aggregate= would be ignored. Drop "
            "aggregate=.",
            id="fieldless_count_refused",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="mean", field="z")
                .encode(x="v", y="w", color=fm.Color(aggregate="mean"))
                .to_svg()
            ),
            _hex_mean_target,
            id="fieldless_equal_mean",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0)
                .encode(x="v", y="w", color=fm.Color(aggregate="count"))
                .to_svg()
            ),
            _hex_count_target,
            id="fieldless_unset_count",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, field="z")
                .encode(x="v", y="w", color=fm.Color(aggregate="mean"))
                .to_svg()
            ),
            _hex_mean_target,
            id="fieldless_unset_mean_with_mark_field",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0)
                .encode(x="v", y="w", color=fm.Color(aggregate="mean"))
                .to_svg()
            ),
            "mark_hex: aggregate='mean' requires field=... (the column to aggregate); "
            "pass field=... or use aggregate='count', which need no field.",
            id="fieldless_unset_mean_needs_field",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0)
                .encode(x="v", y="w", color=fm.Color("z", aggregate="count"))
                .to_svg()
            ),
            _hex_count_target,
            id="named_count_consumed",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="mean", field="w")
                .encode(x="v", y="w", color=fm.Color("z", aggregate="mean"))
                .to_svg()
            ),
            "mark_hex: color='z' is drawn nowhere; the cells are colored by "
            "aggregate='mean' of field='w'. Bind color= to the aggregated column, or "
            "drop it.",
            id="consumed_op_other_field_drawn_nowhere",
        ),
        pytest.param(
            lambda: fm.jointplot(
                _XYZ_ROWS,
                x="v",
                y="w",
                kind="hex",
                encode={"color": fm.Color("z", aggregate="mean")},
            ).to_svg(),
            lambda: fm.jointplot(
                _XYZ_ROWS,
                x="v",
                y="w",
                kind="hex",
                joint_kws={"aggregate": "mean", "field": "z"},
                encode={"color": fm.Color("z", aggregate="mean")},
            ).to_svg(),
            id="jointplot_unset_consumed",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0)
                .encode(x="v", y="w", color=fm.Color("z", aggregate="q1"))
                .to_svg()
            ),
            "mark_hex: aggregate must be one of ['count', 'max', 'mean', 'median', 'min', "
            "'std', 'stdev', 'sum', 'var', 'variance']; got 'q1'",
            id="invalid_op_voiced_as_mark_hex",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="std", field="z")
                .encode(x="v", y="w", color=fm.Color("z", aggregate="stdev"))
                .to_svg()
            ),
            lambda: (
                fm.Chart(_XYZ_ROWS)
                .mark_hex(bin_size=4.0, aggregate="std", field="z")
                .encode(x="v", y="w")
                .to_svg()
            ),
            id="std_stdev_alias_consumed",
        ),
    ],
)
def test_hex_color_aggregate_is_consumed_or_refused(
    build: Callable[[], str], expected: str | Callable[[], str]
):
    """A color ``aggregate=`` equal to ``mark_hex``'s own (or with its own
    ``aggregate=`` unset) is consumed -- byte-identical to the matching
    direct call -- resolved before the color's field is read, so a
    fieldless color's aggregate= folds or refuses too; a differing op is
    refused naming ``mark_hex``'s own ``aggregate=``/``field=``.
    """
    if isinstance(expected, str):
        with pytest.raises(ValueError) as excinfo:
            build()
        assert str(excinfo.value) == expected
    else:
        assert build() == expected()


def test_hex_fold_fixture_tells_mean_sum_and_count_apart():
    """At ``bin_size=4.0`` (more than one row per cell), ``mean``, ``sum``
    and ``count`` render three different SVGs -- the fixture the fold
    rows above compare against can actually discriminate the ops, unlike
    ``mark_hex()``'s own default bin size, which puts one row in each cell
    and renders every op byte-identically.
    """
    mean_svg = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(bin_size=4.0, aggregate="mean", field="z")
        .encode(x="v", y="w")
        .to_svg()
    )
    sum_svg = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(bin_size=4.0, aggregate="sum", field="z")
        .encode(x="v", y="w")
        .to_svg()
    )
    count_svg = fm.Chart(_XYZ_ROWS).mark_hex(bin_size=4.0).encode(x="v", y="w").to_svg()
    assert mean_svg != sum_svg
    assert mean_svg != count_svg
    assert sum_svg != count_svg


def test_mark_hex_aggregate_is_unset_by_default():
    """``mark_hex(aggregate=)``'s default is ``None`` (unset), not
    ``"count"``: the signature now states it, an explicit ``None`` renders
    byte-identically to leaving it out, and the vocabulary is still
    checked at the call for a real value.
    """
    assert inspect.signature(fm.Chart.mark_hex).parameters["aggregate"].default is None

    bare = fm.Chart(_XY_ROWS).mark_hex().encode(x="v", y="w").to_svg()
    explicit_none = fm.Chart(_XY_ROWS).mark_hex(aggregate=None).encode(x="v", y="w").to_svg()
    assert bare == explicit_none
    assert _digest(bare) == "ae014e0549a642546e83a55663c996bcfb78548e077fde3171eb3a29e98dfab5"

    with pytest.raises(ValueError, match=r"mark_hex: aggregate must be one of"):
        fm.Chart(_XY_ROWS).mark_hex(aggregate="geomean").encode(x="v", y="w")


def test_jointplot_hex_invalid_color_op_speaks_as_jointplot():
    """An invalid color ``aggregate=`` consumed through jointplot's hex
    center refuses in jointplot's own voice, not a bare ``mark_hex:``
    naming a call jointplot's caller never wrote -- the same entry every
    other re-voiced hex refusal in this file already speaks as.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.jointplot(
            _XYZ_ROWS, x="v", y="w", kind="hex", encode={"color": fm.Color("z", aggregate="q1")}
        ).to_svg()
    assert str(excinfo.value) == (
        "jointplot: aggregate must be one of ['count', 'max', 'mean', 'median', 'min', "
        "'std', 'stdev', 'sum', 'var', 'variance']; got 'q1'"
    )


def test_hex_color_binned_legend_is_unaffected_by_the_aggregate_fold():
    """A binned color legend (no ``aggregate=`` of its own) still renders its
    own stepped scale: the fold only ever touches a color's ``aggregate=``,
    never its ``bin=``.
    """
    plain = fm.Chart(_XYZ_ROWS).mark_hex(aggregate="mean").encode(x="v", y="w", color="z").to_svg()
    binned = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(aggregate="mean")
        .encode(x="v", y="w", color=fm.Color("z", bin=True))
        .to_svg()
    )
    assert binned != plain
    # The binned legend draws the same whether the field is named directly
    # or supplied by mark_hex's own aggregate=/field=.
    via_field = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(aggregate="mean", field="z")
        .encode(x="v", y="w", color=fm.Color("z", bin=True))
        .to_svg()
    )
    assert binned == via_field


def test_hex_fieldless_consumed_color_styles_the_cells():
    """A fieldless consumed color's own channel properties (``title=``,
    ``scheme=``) style the cells the same as a field-bearing consumed
    color's do -- not only the ``aggregate=``/``field=`` fold itself.
    """
    default_title = (
        fm.Chart(_XY_ROWS)
        .mark_hex()
        .encode(x="v", y="w", color=fm.Color(aggregate="count"))
        .to_svg()
    )
    titled = (
        fm.Chart(_XY_ROWS)
        .mark_hex()
        .encode(x="v", y="w", color=fm.Color(aggregate="count", title="Points"))
        .to_svg()
    )
    assert titled != default_title
    assert "Points" in titled
    assert "Points" not in default_title

    unscyled = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(bin_size=4.0, field="z")
        .encode(x="v", y="w", color=fm.Color(aggregate="mean"))
        .to_svg()
    )
    schemed = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(bin_size=4.0, field="z")
        .encode(x="v", y="w", color=fm.Color(aggregate="mean", scheme="magma"))
        .to_svg()
    )
    assert schemed != unscyled
    assert "#000004" in schemed  # magma's first stop color
    # A field-bearing consumed color's own scheme= draws the identical
    # cells and legend -- the fieldless carryover is not a second,
    # drifting mechanism.
    via_field = (
        fm.Chart(_XYZ_ROWS)
        .mark_hex(bin_size=4.0)
        .encode(x="v", y="w", color=fm.Color("z", aggregate="mean", scheme="magma"))
        .to_svg()
    )
    assert schemed == via_field


def test_mark_raster_and_mark_contour_refuse_a_color_aggregate():
    """Neither mark colors its cells/levels from a chart-level ``color=``;
    a color ``aggregate=`` is always refused, naming each mark's own knob,
    or dropping the kwarg outright when the color names no field.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_XYZ_ROWS).mark_raster().encode(
            x="v", y="w", color=fm.Color("z", aggregate="mean")
        ).to_svg()
    assert str(excinfo.value) == (
        "mark_raster(color=Color('z', aggregate='mean')) is not supported: the cells are "
        "colored by the statistic's own value, so aggregate= would be ignored. Aggregate "
        "the cells with mark_raster(aggregate='mean', field='z'), or drop aggregate=."
    )

    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_XY_ROWS).mark_raster().encode(
            x="v", y="w", color=fm.Color(aggregate="mean")
        ).to_svg()
    assert str(excinfo.value) == (
        "mark_raster(color=Color(aggregate='mean')) is not supported: the cells are "
        "colored by the statistic's own value, so aggregate= would be ignored. Drop "
        "aggregate=."
    )

    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_XYZ_ROWS).mark_contour().encode(
            x="v", y="w", color=fm.Color("z", aggregate="mean")
        ).to_svg()
    assert str(excinfo.value) == (
        "mark_contour(color=Color('z', aggregate='mean')) is not supported: the levels are "
        "colored by the statistic's own value, so aggregate= would be ignored. Set the "
        "levels with mark_contour's thresholds=, or drop aggregate=."
    )


_AUTO_RASTER_ROWS = pl.DataFrame(
    {
        "x": [float(i % 5) for i in range(10)],
        "y": [float(i % 3) for i in range(10)],
        "c": [float(i) for i in range(10)],
    }
)


@pytest.mark.parametrize(
    "encode_kwargs,expect_image,warning_match,compare_to_unthresholded",
    [
        pytest.param(
            {"x": X("x:Q", bin=True), "y": "y:Q"},
            False,
            "may produce large output",
            True,
            id="x_bin",
        ),
        pytest.param(
            {"x": "x:Q", "y": Y("y:Q", aggregate="mean")},
            False,
            None,
            True,
            id="y_aggregate",
        ),
        pytest.param(
            {"x": X("x:Q", aggregate="mean"), "y": "y:Q"},
            False,
            None,
            True,
            id="x_aggregate",
        ),
        pytest.param(
            {"x": "x:Q", "y": Y("y:Q", bin=True)},
            False,
            "may produce large output",
            True,
            id="y_bin",
        ),
        pytest.param(
            {"x": X("x:Q", bin=True), "y": Y("y:Q", aggregate="mean")},
            False,
            None,
            True,
            id="bin_and_aggregate_together",
        ),
        pytest.param(
            {"x": X("x:Q", bin=False), "y": "y:Q"},
            True,
            "substituted mark_raster",
            False,
            id="x_bin_false_still_substitutes",
        ),
        pytest.param(
            {"x": X("x:Q", aggregate=None), "y": "y:Q"},
            True,
            "substituted mark_raster",
            False,
            id="x_aggregate_none_still_substitutes",
        ),
        pytest.param(
            {"x": "x:Q", "y": "y:Q"},
            True,
            "substituted mark_raster",
            False,
            id="plain_control",
        ),
    ],
)
def test_auto_raster_never_substitutes_a_binned_or_aggregated_position(
    encode_kwargs: dict,
    expect_image: bool,
    warning_match: str | None,
    compare_to_unthresholded: bool,
):
    """A binned or aggregated x/y draws per row or per group, never a
    position the raster can draw: auto-raster excludes it (silently for an
    aggregate, with the large-output warning for a bin, like the missing/
    non-quantitative gate) rather than substituting a raster that would
    silently drop it, rendering byte-equal to the same chart with
    auto-raster disabled outright. ``x_aggregate`` and ``y_bin`` pin the
    other operand of each ``or`` the exclusion checks; ``bin_and_aggregate_together``
    pins that the aggregate exclusion runs first and wins silently, with no
    large-output warning, when both are present on different channels.
    ``x_bin_false_still_substitutes`` and ``x_aggregate_none_still_substitutes``
    pin that the unset spellings (``bin=False``, ``aggregate=None``) are not
    treated as a binned/aggregated position -- they substitute exactly like
    the plain quantitative control. The plain quantitative control is
    unaffected by this exclusion: it still substitutes.
    """
    import warnings as _warnings

    over_threshold_cfg = fm.RenderConfig(raster_threshold=0)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        over_threshold = (
            fm.Chart(_AUTO_RASTER_ROWS)
            .mark_point()
            .encode(**encode_kwargs)
            .properties(render_config=over_threshold_cfg)
            .to_svg()
        )
    messages = [str(w.message) for w in caught]
    if warning_match is None:
        assert not any("large output" in m or "substituted" in m for m in messages), messages
    else:
        assert any(warning_match in m for m in messages), messages

    assert ("<image" in over_threshold) == expect_image
    if compare_to_unthresholded:
        unthresholded = (
            fm.Chart(_AUTO_RASTER_ROWS)
            .mark_point()
            .encode(**encode_kwargs)
            .properties(render_config=fm.RenderConfig(raster_threshold=None))
            .to_svg()
        )
        assert over_threshold == unthresholded


@pytest.mark.parametrize(
    "encode_kwargs",
    [
        pytest.param({"color": fm.Color("c:Q", aggregate="sum")}, id="color_aggregate"),
        pytest.param({"size": fm.Size("c:Q", aggregate="mean")}, id="size_aggregate"),
        pytest.param({"opacity": fm.Opacity("c:Q", aggregate="mean")}, id="opacity_aggregate"),
    ],
)
def test_auto_raster_never_substitutes_any_channels_own_aggregate(encode_kwargs: dict):
    """A non-positional channel's own ``aggregate=`` (color, size, opacity)
    excludes auto-raster and warns with the channel-specific text, naming
    only the remedy that renders -- never ``.mark_raster()``, which refuses
    a color aggregate= and fails raw on size/opacity.
    """
    import warnings as _warnings

    channel = next(iter(encode_kwargs))
    over_threshold_cfg = fm.RenderConfig(raster_threshold=0)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        over_threshold = (
            fm.Chart(_AUTO_RASTER_ROWS)
            .mark_point()
            .encode(x="x:Q", y="y:Q", **encode_kwargs)
            .properties(render_config=over_threshold_cfg)
            .to_svg()
        )
    messages = [str(w.message) for w in caught]
    assert messages == [
        f"Chart has 10 marks which may produce large output, and its {channel} "
        "encoding carries its own aggregate=, so its marks are drawn per group -- "
        "`.mark_raster()` cannot draw it either. Compute the aggregate with "
        "`.transform(fm.transform_aggregate(...))` and bind the output column "
        "with no aggregate= of its own, or set `raster=False` to suppress "
        "this warning."
    ]
    assert "<image" not in over_threshold

    unthresholded = (
        fm.Chart(_AUTO_RASTER_ROWS)
        .mark_point()
        .encode(x="x:Q", y="y:Q", **encode_kwargs)
        .properties(render_config=fm.RenderConfig(raster_threshold=None))
        .to_svg()
    )
    assert over_threshold == unthresholded


@pytest.mark.parametrize(
    "channel,Channel",
    [
        pytest.param("color", fm.Color, id="color"),
        pytest.param("size", fm.Size, id="size"),
        pytest.param("opacity", fm.Opacity, id="opacity"),
    ],
)
def test_warn_channel_aggregate_excluded_remedy_renders(channel: str, Channel: type):
    """Both remedies ``_warn_channel_aggregate_excluded`` names actually
    render: pre-aggregating with ``.transform(fm.transform_aggregate(...))``
    and binding the output column with no ``aggregate=`` of its own (which
    substitutes), and ``raster=False`` (which renders per element, with no
    exclusion warning either way) -- for color, size and opacity alike.
    """
    import warnings as _warnings

    chart = (
        fm.Chart(_AUTO_RASTER_ROWS)
        .mark_point()
        .transform(
            fm.transform_aggregate({"field": "c", "fn": "mean", "as": "mean_c"}, groupby=["x", "y"])
        )
        .encode(x="x:Q", y="y:Q", **{channel: Channel("mean_c:Q")})
        .properties(render_config=fm.RenderConfig(raster_threshold=0))
    )
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        svg = chart.to_svg()
    assert "<image" in svg
    assert not any("may produce large output" in str(w.message) for w in caught)

    # raster=False on the EXCLUDED (still-aggregated) chart suppresses the
    # exclusion warning and renders per element.
    excluded_chart = (
        fm.Chart(_AUTO_RASTER_ROWS)
        .mark_point()
        .encode(x="x:Q", y="y:Q", **{channel: Channel("c:Q", aggregate="mean")})
        .properties(render_config=fm.RenderConfig(raster_threshold=0))
    )
    with _warnings.catch_warnings(record=True) as caught_suppressed:
        _warnings.simplefilter("always")
        excluded_chart.to_svg(raster=False)
    assert caught_suppressed == []


def test_warn_bin_excluded_names_only_remedies_that_render():
    """The positional bin= exclusion's warning text is pinned exactly, and
    each remedy it names actually renders: an aggregate= on the OPPOSING
    positional channel (not the same channel -- a single channel's own
    bin= and aggregate= together is refused at resolution, and
    `.mark_raster()` refuses the same bin= too, so neither is named)."""
    import warnings as _warnings

    x_bin_cfg = fm.RenderConfig(raster_threshold=0)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        fm.Chart(_AUTO_RASTER_ROWS).mark_point().encode(x=X("x:Q", bin=True), y="y:Q").properties(
            render_config=x_bin_cfg
        ).to_svg()
    messages = [str(w.message) for w in caught]
    assert messages == [
        "Chart has 10 marks which may produce large output, and its x or y "
        "encoding carries a bin= with no aggregate=, so `.mark_raster()` "
        "cannot draw it either. Put an aggregate= on the opposing "
        "positional channel (e.g. `y=fm.Y('field', aggregate='mean')` "
        "beside `x=fm.X('field', bin=True)`), or set `raster=False` to "
        "suppress this warning."
    ]

    # Remedy 1: an aggregate= on the OPPOSING positional channel renders.
    fm.Chart(_AUTO_RASTER_ROWS).mark_point().encode(
        x=X("x:Q", bin=True), y=Y("y:Q", aggregate="mean")
    ).properties(render_config=x_bin_cfg).to_svg()

    # Remedy 2: raster=False renders with no warning at all.
    with _warnings.catch_warnings(record=True) as caught_suppressed:
        _warnings.simplefilter("always")
        fm.Chart(_AUTO_RASTER_ROWS).mark_point().encode(x=X("x:Q", bin=True), y="y:Q").to_svg(
            raster=False
        )
    assert caught_suppressed == []

    # The remedy the warning does NOT name (the same channel's own bin=
    # and aggregate=) is refused, confirming it would be dishonest to
    # suggest.
    with pytest.raises(
        ValueError, match="cannot be both the bin position and the aggregated value"
    ):
        fm.Chart(_AUTO_RASTER_ROWS).mark_point().encode(
            x=X("x:Q", bin=True, aggregate="mean"), y="y:Q"
        ).properties(render_config=x_bin_cfg).to_svg()

    # `.mark_raster()` itself still refuses the same bin=, confirming that
    # remedy would be dishonest too.
    with pytest.raises(ValueError, match="Set the grid with mark_raster's resolution="):
        fm.Chart(_AUTO_RASTER_ROWS).mark_raster().encode(x=X("x:Q", bin=True), y="y:Q").to_svg()


@pytest.mark.parametrize("mark", ["mark_hex", "mark_raster", "mark_contour"])
def test_a_wrong_type_color_bin_is_the_existing_channel_validation_error(mark: str):
    """A color ``bin=`` of the wrong type is refused by the channel itself,
    at construction -- before mark_hex/mark_raster/mark_contour (whose own
    refusals never touch a color's ``bin=``) ever see it, reached through
    each mark's own chain.
    """
    with pytest.raises(ValueError, match=r"Color\(bin=123\) is not a bin"):
        getattr(fm.Chart(_XYZ_ROWS), mark)().encode(
            x="v", y="w", color=fm.Color("z", bin=123)
        ).to_svg()


@pytest.mark.parametrize(
    "mark,reason,remedy",
    [
        pytest.param(
            "mark_density",
            "the kernel density estimate reads raw rows",
            "Bin the field per row with .transform(fm.Bin('v', per_row=True)) and bind its "
            "'bin_start' column, or drop bin=.",
            id="mark_density",
        ),
        pytest.param(
            "mark_contour",
            "the kernel density estimate reads raw rows",
            "Bin the field per row with .transform(fm.Bin('v', per_row=True)) and bind its "
            "'bin_start' column, or drop bin=.",
            id="mark_contour",
        ),
        pytest.param(
            "mark_hex",
            "the hexagons bin x and y by their own grid",
            "Size the cells with mark_hex's bin_size=, or drop bin=.",
            id="mark_hex",
        ),
        pytest.param(
            "mark_raster",
            "the raster bins x and y onto its own pixel grid",
            "Set the grid with mark_raster's resolution=, or drop bin=.",
            id="mark_raster",
        ),
        pytest.param(
            "mark_qq",
            "the Q-Q plot reads the raw sample",
            "Bin the field per row with .transform(fm.Bin('v', per_row=True)) and bind its "
            "'bin_start' column, or drop bin=.",
            id="mark_qq",
        ),
    ],
)
def test_a_wrong_type_positional_bin_takes_each_marks_statistic_refusal(
    mark: str, reason: str, remedy: str
):
    """A positional ``X``/``Y`` has no construction-time ``bin=`` type check
    (unlike a color-family channel): ``X('v', bin='foo')`` constructs fine
    and reaches each statistic mark's own x/y refusal, exactly as it does
    on ``mark_density``.
    """
    encode_kwargs = {"x": X("v", bin="foo")}
    if mark != "mark_qq":
        encode_kwargs["y"] = "w"
    with pytest.raises(ValueError) as excinfo:
        getattr(fm.Chart(_XY_ROWS), mark)().encode(**encode_kwargs).to_svg()
    assert str(excinfo.value) == (
        f"{mark}(x=X('v', bin='foo')) is not supported: {reason}, so bin= would be "
        f"ignored. {remedy}"
    )


#: A frame with a caller's own ``sample`` column -- the QQ transform's own
#: output column shares that name by coincidence, not by provenance, so a
#: column named ``"sample"`` is refused exactly like any other field.
_QQ_SAMPLE_ROWS = _HIST_ROWS.with_columns(sample=pl.col("v") * 2.0)


@pytest.mark.parametrize(
    "rows,y",
    [
        pytest.param(_HIST_ROWS, "w", id="w"),
        pytest.param(_QQ_SAMPLE_ROWS, "sample", id="sample_column"),
        pytest.param(_QQ_SAMPLE_ROWS, Y("sample", title="Sample quantiles"), id="sample_titled"),
        pytest.param(_QQ_SAMPLE_ROWS, Y("sample", bin=True), id="sample_bin"),
        pytest.param(_QQ_SAMPLE_ROWS, Y("sample", aggregate="mean"), id="sample_aggregate"),
    ],
)
def test_mark_qq_refuses_a_field_bearing_y(rows: pl.DataFrame, y: Any):
    """``mark_qq`` draws its own sample quantiles on y, ignoring whatever
    ``y=`` binds, so a field-bearing one is refused like ``displot``'s
    ``kind='ecdf'`` -- the same message shape, mark_qq's own voice and
    knob -- with no exemption: a column named ``"sample"`` (the caller's
    own, or named via ``bin=``/``aggregate=``, checked after the field
    itself) is refused exactly like any other field, since a column name
    is not a provenance.
    """
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(rows).mark_qq().encode(x="v", y=y).to_svg()
    bound = y.field if isinstance(y, Y) else y
    assert str(excinfo.value) == (
        f"mark_qq(y={bound!r}) is not supported -- mark_qq draws the sample quantiles of "
        "x= on y, so a second variable has nowhere to go. Drop y=, or style the axis with "
        "fm.Y(title=...) and no field."
    )


def test_mark_qq_fieldless_y_title_styles_the_axis():
    """A fieldless ``Y(title=...)`` is the door ``mark_qq``'s field-bearing-y
    refusal names: it renders, carries the title into the SVG, and differs
    from the untitled chart.
    """
    titled = fm.Chart(_HIST_ROWS).mark_qq().encode(x="v", y=Y(title="Sample quantiles")).to_svg()
    plain = fm.Chart(_HIST_ROWS).mark_qq().encode(x="v").to_svg()
    assert titled != plain
    assert "Sample quantiles" in titled


@pytest.mark.parametrize(
    "with_kwarg,without",
    [
        (
            lambda: fm.Chart(_XY_ROWS).mark_contour().encode(x=X("v", bin=None), y="w").to_svg(),
            lambda: fm.Chart(_XY_ROWS).mark_contour().encode(x="v", y="w").to_svg(),
        ),
        (
            lambda: (
                fm.Chart(_XY_ROWS).mark_contour().encode(x=X("v", aggregate=None), y="w").to_svg()
            ),
            lambda: fm.Chart(_XY_ROWS).mark_contour().encode(x="v", y="w").to_svg(),
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x=X("v", bin=False), y="w").to_svg(),
            lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x="v", y="w").to_svg(),
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x=X("v", aggregate=None), y="w").to_svg(),
            lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x="v", y="w").to_svg(),
        ),
        (
            lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x=X("v", bin=None), y="w").to_svg(),
            lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x="v", y="w").to_svg(),
        ),
        (
            lambda: (
                fm.Chart(_XY_ROWS).mark_raster().encode(x=X("v", aggregate=None), y="w").to_svg()
            ),
            lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x="v", y="w").to_svg(),
        ),
        (
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x=X("v", bin=False)).to_svg(),
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v").to_svg(),
        ),
        (
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x=X("v", aggregate=None)).to_svg(),
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v").to_svg(),
        ),
        (
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v", y=Y(bin=None)).to_svg(),
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v").to_svg(),
        ),
        (
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v", y=Y(aggregate=None)).to_svg(),
            lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v").to_svg(),
        ),
    ],
    ids=[
        "contour_x_bin_none",
        "contour_x_aggregate_none",
        "hex_x_bin_false",
        "hex_x_aggregate_none",
        "raster_x_bin_none",
        "raster_x_aggregate_none",
        "qq_x_bin_false",
        "qq_x_aggregate_none",
        "qq_y_bin_none",
        "qq_y_aggregate_none",
    ],
)
def test_bin_or_aggregate_unset_values_render_byte_identically_to_leaving_them_out(
    with_kwarg: Callable[[], str], without: Callable[[], str]
):
    """``bin=None``, ``bin=False`` and ``aggregate=None`` ask for nothing, so
    the statistic renders exactly as it would with neither kwarg at all --
    the refusals above never even see them (:func:`~ferrum.encoding.base._unset_statistic`).
    """
    assert with_kwarg() == without()


#: Two rows per ``v``, so an aggregated ``y``/``y2`` (``min``/``max`` over
#: each group) differs from the raw ``w`` a group key would otherwise
#: read verbatim (a single row per ``v`` makes ``min(w) == max(w) == w``,
#: which cannot discriminate aggregation from the plain render).
_RIBBON_TWO_PER_V_ROWS = pl.DataFrame(
    {"v": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "w": [1.0, 3.0, 2.0, 6.0, 1.0, 5.0]}
)


def test_mark_ribbon_honors_bin_and_aggregate_never_refused():
    """``mark_ribbon`` lowers as one unlayered mark, like ``mark_area`` --
    it computes no statistic of its own, so a channel ``bin=``/``aggregate=``
    is honored exactly as a plain mark's is, never refused. Each half first
    renders without raising, then is compared against a plain ribbon
    carrying that SAME render's own axis title, so only the band's
    geometry -- never the title text -- can make the two differ.
    """
    aggregated = (
        fm.Chart(_RIBBON_TWO_PER_V_ROWS)
        .mark_ribbon()
        .encode(x="v", y=Y("w", aggregate="min"), y2=Y("w", aggregate="max"))
        .to_svg()
    )
    plain_same_title = (
        fm.Chart(_RIBBON_TWO_PER_V_ROWS)
        .mark_ribbon()
        .encode(x="v", y=Y("w", title=y_axis_title(aggregated)), y2="w")
        .to_svg()
    )
    assert aggregated != plain_same_title

    binned = (
        fm.Chart(_RIBBON_TWO_PER_V_ROWS)
        .mark_ribbon()
        .encode(x=X("v", bin=True), y="w", y2="w")
        .to_svg()
    )
    plain_same_x_title = (
        fm.Chart(_RIBBON_TWO_PER_V_ROWS)
        .mark_ribbon()
        .encode(x=X("v", title=x_axis_title(binned)), y="w", y2="w")
        .to_svg()
    )
    assert binned != plain_same_x_title


@pytest.mark.parametrize(
    "build_plain,build_with_kwarg",
    [
        pytest.param(
            lambda rows: fm.Chart(rows).mark_boxplot().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_boxplot().encode(x="cat", y=Y("v", bin=True)),
            id="boxplot-y_bin",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_boxplot().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_boxplot().encode(x="cat", y=Y("v", aggregate="mean")),
            id="boxplot-y_aggregate",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_boxen().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_boxen().encode(x="cat", y=Y("v", bin=True)),
            id="boxen-y_bin",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_boxen().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_boxen().encode(x="cat", y=Y("v", aggregate="mean")),
            id="boxen-y_aggregate",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_violin().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_violin().encode(x="cat", y=Y("v", bin=True)),
            id="violin-y_bin",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_violin().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_violin().encode(x="cat", y=Y("v", aggregate="mean")),
            id="violin-y_aggregate",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_errorbar().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_errorbar().encode(x="cat", y=Y("v", bin=True)),
            id="errorbar-y_bin",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_errorbar().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_errorbar().encode(x="cat", y=Y("v", aggregate="mean")),
            id="errorbar-y_aggregate",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_swarm().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_swarm().encode(x="cat", y=Y("v", bin=True)),
            id="swarm-y_bin",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_swarm().encode(x="cat", y="v"),
            lambda rows: fm.Chart(rows).mark_swarm().encode(x="cat", y=Y("v", aggregate="mean")),
            id="swarm-y_aggregate",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_errorband().encode(x="v", y="w"),
            lambda rows: fm.Chart(rows).mark_errorband().encode(x="v", y=Y("w", bin=True)),
            id="errorband-y_bin",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_errorband().encode(x="v", y="w"),
            lambda rows: fm.Chart(rows).mark_errorband().encode(x="v", y=Y("w", aggregate="mean")),
            id="errorband-y_aggregate",
        ),
        pytest.param(
            lambda rows: fm.Chart(rows).mark_errorband().encode(x="v", y="w"),
            lambda rows: fm.Chart(rows).mark_errorband().encode(x=X("v", bin=True), y="w"),
            id="errorband-x_bin",
        ),
    ],
)
def test_categorical_composites_bin_and_aggregate_stay_silently_ignored(
    build_plain: Callable[[pl.DataFrame], fm.Chart],
    build_with_kwarg: Callable[[pl.DataFrame], fm.Chart],
):
    """boxplot, boxen, violin, errorbar, swarm and errorband share no prep
    with contour/hex/raster/qq's x/y refusals, so a channel ``bin=``/
    ``aggregate=`` on one of them stays silently ignored, byte-identical
    to the same chart without it.
    """
    rows = _XY_ROWS.with_columns(cat=pl.lit("a"))
    assert build_plain(rows).to_svg() == build_with_kwarg(rows).to_svg()


@pytest.mark.parametrize(
    "name,build",
    [
        (
            "contour_fill_false",
            lambda: fm.Chart(_XY_ROWS).mark_contour(fill=False).encode(x="v", y="w").to_svg(),
        ),
        (
            "contour_fill_true",
            lambda: fm.Chart(_XY_ROWS).mark_contour(fill=True).encode(x="v", y="w").to_svg(),
        ),
        ("hex", lambda: fm.Chart(_XY_ROWS).mark_hex().encode(x="v", y="w").to_svg()),
        ("raster", lambda: fm.Chart(_XY_ROWS).mark_raster().encode(x="v", y="w").to_svg()),
        ("qq", lambda: fm.Chart(_HIST_ROWS).mark_qq().encode(x="v").to_svg()),
    ],
)
def test_every_mark_without_the_new_kwargs_renders_unchanged(name: str, build: Callable[[], str]):
    """None of the four marks' plain renders (no ``bin=``/``aggregate=`` at
    all) moved a byte: each is pinned by its own digest below.
    """
    digests = {
        "contour_fill_false": "051c99e0a602a1ad450a10ec8dd98273b29f279dde51bbb762e57bfb8b92a04b",
        "contour_fill_true": "be0c088a81956d6b76a03f7e7d5928e10f9e49485a0483695288c8f8be070e76",
        "hex": "ae014e0549a642546e83a55663c996bcfb78548e077fde3171eb3a29e98dfab5",
        "raster": "4c653a0f4debd75b93a15bc960155b802c9235b16000168b8086f217da32148e",
        "qq": "f512624b5ced12463225b5cc0cb3519c42285058bd29b091f7256ad9fdec1a59",
    }
    assert _digest(build()) == digests[name]


# ---------------------------------------------------------------------------
# A declared categorical type (N/O) on an axis a statistic reads as numbers.
#
# Every statistic that already refuses a bin=/aggregate= it cannot honor
# (above) also refuses a declared `N`/`O` on the same axes, with its own
# template -- the same rule Chart.encode's shorthand parser applies: a
# declared type is a real instruction, not a hint runtime dtype can
# override, and a statistic that reads a channel as numbers cannot place a
# category on it. Each expected message below is written independently of
# the implementation, from the statistic's own reason and the shared
# "Drop the type (...)" ending, so a wrong field, a wrong type code, or a
# changed reason would fail the comparison.
# ---------------------------------------------------------------------------


def _fit_categorical_text(channel: str, field: str, type_: str) -> str:
    return (
        f"the fit reads {channel} as numbers and draws its curve along a continuous "
        f"axis, so type={type_!r} would place the fitted values as categories. "
        f"Drop the type ({channel}={field!r})."
    )


def _kde_categorical_text(channel: str, field: str, type_: str) -> str:
    return (
        f"the kernel density estimate reads {channel} as numbers, so type={type_!r} "
        f"would place the rows as categories. Drop the type ({channel}={field!r})."
    )


def _hex_categorical_text(channel: str, field: str, type_: str) -> str:
    return (
        f"the hexagons bin {channel} by their own grid, so type={type_!r} would "
        f"place the rows as categories. Drop the type ({channel}={field!r})."
    )


def _raster_categorical_text(channel: str, field: str, type_: str) -> str:
    return (
        f"the raster bins {channel} onto its own pixel grid, so type={type_!r} "
        f"would place the rows as categories. Drop the type ({channel}={field!r})."
    )


def _qq_categorical_text(channel: str, field: str, type_: str) -> str:
    return (
        f"the Q-Q plot reads {channel} as the raw sample, so type={type_!r} would "
        f"place the rows as categories. Drop the type ({channel}={field!r})."
    )


def _ecdf_categorical_text(channel: str, field: str, type_: str) -> str:
    return (
        f"the ECDF counts the raw values of {channel}, so type={type_!r} would "
        f"place the rows as categories. Drop the type ({channel}={field!r})."
    )


def _histogram_categorical_text(channel: str, field: str, type_: str, value_axis: str) -> str:
    return (
        f"the histogram bins {channel} as numbers and draws a bar per bin, so "
        f"type={type_!r} would draw one bar per row instead of one per category. To "
        f"count each category, use mark_bar().encode({channel}={field + ':' + type_!r}, "
        f"{value_axis}='count()')."
    )


def _categorical_message(
    voice: str, channel: str, cls: str, field: str, type_: str, body: str
) -> str:
    return f"{voice}({channel}={cls}({field!r}, type={type_!r})) is not supported: {body}"


@dataclass(frozen=True)
class _CategoricalSite:
    """One statistic's own axis and reads clause for the categorical-type grid."""

    build: Callable[[str], fm.Chart]
    voice: str
    channel: str
    cls: str
    field: str
    body: Callable[[str, str, str], str]


_CATEGORICAL_SITES: dict[str, _CategoricalSite] = {
    "histogram_x": _CategoricalSite(
        lambda t: fm.Chart(_HIST_ROWS).mark_histogram().encode(x=f"v{t}"),
        "mark_histogram",
        "x",
        "X",
        "v",
        lambda ch, f, t: _histogram_categorical_text(ch, f, t, "y"),
    ),
    "histogram_horizontal_y": _CategoricalSite(
        lambda t: fm.Chart(_HIST_ROWS).mark_histogram(orientation="horizontal").encode(y=f"v{t}"),
        "mark_histogram",
        "y",
        "Y",
        "v",
        lambda ch, f, t: _histogram_categorical_text(ch, f, t, "x"),
    ),
    "density_x": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_density().encode(x=f"v{t}"),
        "mark_density",
        "x",
        "X",
        "v",
        _kde_categorical_text,
    ),
    "density_bivariate_y": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_density().encode(x="v", y=f"w{t}"),
        "mark_density",
        "y",
        "Y",
        "w",
        _kde_categorical_text,
    ),
    "contour_x": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_contour().encode(x=f"v{t}", y="w"),
        "mark_contour",
        "x",
        "X",
        "v",
        _kde_categorical_text,
    ),
    "contour_y": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_contour().encode(x="v", y=f"w{t}"),
        "mark_contour",
        "y",
        "Y",
        "w",
        _kde_categorical_text,
    ),
    "hex_x": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_hex().encode(x=f"v{t}", y="w"),
        "mark_hex",
        "x",
        "X",
        "v",
        _hex_categorical_text,
    ),
    "hex_y": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_hex().encode(x="v", y=f"w{t}"),
        "mark_hex",
        "y",
        "Y",
        "w",
        _hex_categorical_text,
    ),
    "raster_x": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_raster().encode(x=f"v{t}", y="w"),
        "mark_raster",
        "x",
        "X",
        "v",
        _raster_categorical_text,
    ),
    "raster_y": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_raster().encode(x="v", y=f"w{t}"),
        "mark_raster",
        "y",
        "Y",
        "w",
        _raster_categorical_text,
    ),
    "qq_x": _CategoricalSite(
        lambda t: fm.Chart(_HIST_ROWS).mark_qq().encode(x=f"v{t}"),
        "mark_qq",
        "x",
        "X",
        "v",
        _qq_categorical_text,
    ),
    "smooth_x": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_smooth().encode(x=f"v{t}", y="w"),
        "mark_smooth",
        "x",
        "X",
        "v",
        _fit_categorical_text,
    ),
    "smooth_y": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_smooth().encode(x="v", y=f"w{t}"),
        "mark_smooth",
        "y",
        "Y",
        "w",
        _fit_categorical_text,
    ),
    "point_then_smooth_x": _CategoricalSite(
        lambda t: fm.Chart(_XY_ROWS).mark_point().mark_smooth().encode(x=f"v{t}", y="w"),
        "mark_smooth",
        "x",
        "X",
        "v",
        _fit_categorical_text,
    ),
    "displot_hist_x": _CategoricalSite(
        lambda t: fm.displot(_HIST_ROWS, x=f"v{t}", kind="hist"),
        "displot",
        "x",
        "X",
        "v",
        lambda ch, f, t: _histogram_categorical_text(ch, f, t, "y"),
    ),
    "displot_kde_x": _CategoricalSite(
        lambda t: fm.displot(_XY_ROWS, x=f"v{t}", y="w", kind="kde"),
        "displot",
        "x",
        "X",
        "v",
        _kde_categorical_text,
    ),
    "displot_kde_y": _CategoricalSite(
        lambda t: fm.displot(_XY_ROWS, x="v", y=f"w{t}", kind="kde"),
        "displot",
        "y",
        "Y",
        "w",
        _kde_categorical_text,
    ),
    "displot_ecdf_x": _CategoricalSite(
        lambda t: fm.displot(_HIST_ROWS, x=f"v{t}", kind="ecdf"),
        "displot",
        "x",
        "X",
        "v",
        _ecdf_categorical_text,
    ),
}


@pytest.mark.parametrize("site_id", list(_CATEGORICAL_SITES), ids=list(_CATEGORICAL_SITES))
@pytest.mark.parametrize("type_", ["N", "O"])
def test_a_categorical_type_on_a_statistics_input_axis_is_refused(site_id: str, type_: str):
    """Every statistic that reads an axis as numbers refuses a declared
    ``N``/``O`` there, with the same message the implementation builds
    independently reconstructed here.
    """
    site = _CATEGORICAL_SITES[site_id]
    with pytest.raises(ValueError) as excinfo:
        site.build(f":{type_}").to_svg()
    assert str(excinfo.value) == _categorical_message(
        site.voice,
        site.channel,
        site.cls,
        site.field,
        type_,
        site.body(site.channel, site.field, type_),
    )


@pytest.mark.parametrize("site_id", list(_CATEGORICAL_SITES), ids=list(_CATEGORICAL_SITES))
@pytest.mark.parametrize("type_", ["", ":Q"])
def test_a_quantitative_or_unset_type_is_honored_on_every_categorical_check_site(
    site_id: str, type_: str
):
    """A negative control for every site above: no type at all, or an
    explicit ``:Q``, renders byte-identical to the bare field -- neither is
    refused, because both are the continuous reading every one of these
    statistics already draws.
    """
    site = _CATEGORICAL_SITES[site_id]
    bare = site.build("").to_svg()
    assert site.build(type_).to_svg() == bare


#: Every _CATEGORICAL_SITES entry whose statistic redraws this axis from
#: its own numeric output (a Bin's column, a KDE grid, a fit's curve),
#: so a declared T never reaches the render -- every site except the two
#: whose axis is the caller's raw column, drawn directly: `point_then_smooth_x`'s
#: overlaid point mark, and `displot_ecdf_x`'s x, which `_desugar.py` never
#: re-points onto a statistic's output column. Those two keep the declared T
#: and are pinned by the test below this one, not here.
_REDRAWN_AXIS_SITES = [
    site_id
    for site_id in _CATEGORICAL_SITES
    if site_id not in ("point_then_smooth_x", "displot_ecdf_x")
]


@pytest.mark.parametrize("site_id", _REDRAWN_AXIS_SITES, ids=_REDRAWN_AXIS_SITES)
def test_a_declared_temporal_type_is_not_applied_on_a_redrawn_statistic_axis(site_id: str):
    """``T`` is not a categorical type either, but every site here redraws
    this axis from its own numeric output, so the declared type never
    reaches the render: the chart renders exactly as the bare field does.
    """
    site = _CATEGORICAL_SITES[site_id]
    bare = site.build("").to_svg()
    assert site.build(":T").to_svg() == bare


def test_a_declared_temporal_type_is_applied_where_the_axis_is_drawn_directly():
    """The two sites excluded above: a declared T changes the render here,
    since the channel that draws the axis is the caller's raw column, not
    a statistic's redrawn output -- the same rule a drawn scatter layer
    (regplot/lmplot, see ``tests/test_regression_keyword_reading.py``) and
    residplot's x follow.
    """
    point_then_smooth = _CATEGORICAL_SITES["point_then_smooth_x"]
    assert point_then_smooth.build(":T").to_svg() != point_then_smooth.build("").to_svg()

    ecdf = _CATEGORICAL_SITES["displot_ecdf_x"]
    assert ecdf.build(":T").to_svg() != ecdf.build("").to_svg()


def test_a_categorical_composite_is_unaffected_by_the_categorical_input_rule():
    """boxplot, boxen, violin, errorbar, swarm and errorband share no prep
    with the statistics above, so a declared categorical type on their own
    category axis -- which is what those marks need it to be -- is drawn,
    never refused by this rule, and each render is pinned by its own
    digest below, computed on the baseline tree.
    """
    rows = _XY_ROWS.with_columns(cat=pl.lit("a"))
    digests = {
        "boxplot": "5f60ffae53d7c230179e3949a64d5634b58d12dfe6c490b8e1c7091619058c77",
        "boxen": "32b2d92d580bfe1fefd4dd67ffa6119822c7c1491ecf0eb0d3bdb2a39efed056",
        "violin": "7af02e47dec63752ac6b1f6307b599a2c97b2907838d3c7b97438efe4e5650e1",
        "errorbar": "0f03b9b9bd8af0fc456ef855c4380348aedf760116c083f87b8afcdecdb65724",
        "swarm": "72f68c4c690be0c64fa0c9ff171755d95df375692130bb3a2371e5730f58667a",
        "errorband": "32b135792eb9da94d17fdbd6e2263e21a35943c5936d34c7fdd79e316e09d193",
    }
    builds: dict[str, Callable[[], fm.Chart]] = {
        "boxplot": lambda: fm.Chart(rows).mark_boxplot().encode(x="cat:N", y="v"),
        "boxen": lambda: fm.Chart(rows).mark_boxen().encode(x="cat:N", y="v"),
        "violin": lambda: fm.Chart(rows).mark_violin().encode(x="cat:N", y="v"),
        "errorbar": lambda: fm.Chart(rows).mark_errorbar().encode(x="cat:N", y="v"),
        "swarm": lambda: fm.Chart(rows).mark_swarm().encode(x="cat:N", y="v"),
        "errorband": lambda: fm.Chart(rows).mark_errorband().encode(x="cat:N", y="v"),
    }
    for name, build in builds.items():
        assert _digest(build().to_svg()) == digests[name], name


def test_a_fieldless_typed_channel_is_left_to_the_statistics_own_rules():
    """A typed channel with no field of its own never names an input
    column, so the categorical-input rule leaves it alone entirely: a
    computed axis styled with a fieldless type renders as the baseline
    does, and a required axis bound fieldless falls through to the
    statistic's own missing-field refusal, never a categorical one that
    would misname the field ``None``.
    """
    density_svg = fm.Chart(_XY_ROWS).mark_density().encode(x="v", y=fm.Y(type="N")).to_svg()
    assert (
        _digest(density_svg) == "45cd0e4ebcfcfa76ea4910c081e3abdf571f0271e335cc5b638fcc7c739ae6f9"
    )

    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_XY_ROWS).mark_smooth().encode(x="v", y=fm.Y(type="N")).to_svg()
    assert str(excinfo.value) == "mark_smooth() requires .encode(x=..., y=...)"

    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_HIST_ROWS).mark_histogram().encode(x=X(type="N")).to_svg()
    assert (
        str(excinfo.value)
        == "mark_histogram() requires .encode(x=...) to specify the histogram field"
    )

    with pytest.raises(ValueError) as excinfo:
        fm.Chart(_HIST_ROWS).mark_qq().encode(x=X(type="N")).to_svg()
    assert str(excinfo.value) == "mark_qq() requires .encode(x=...) to specify the sample field"


@pytest.mark.parametrize("kind", ["scatter", "kde", "hist", "hex", "reg"])
def test_a_marginal_histograms_categorical_refusal_ends_in_drop_the_type(kind: str):
    """jointplot's marginal histogram re-voices the same categorical refusal
    mark_histogram raises directly, but jointplot's ``JointChart`` result
    has no ``.encode()``/``mark_bar()`` to chain, so it ends in the fit
    family's "Drop the type" ending instead of the count-by-category
    rewrite -- for every center ``kind=``, since the marginal that refuses
    is the same ``marginal_kind="hist"`` default regardless of the center.
    """
    rows = pl.DataFrame(
        {"v": [float(i) for i in range(1, 21)], "w": [float(i) * 2.0 + 1.0 for i in range(1, 21)]}
    )
    with pytest.raises(ValueError) as excinfo:
        fm.jointplot(rows, x="v:N", y="w", kind=kind).to_svg()
    assert str(excinfo.value) == (
        "jointplot(x=X('v', type='N')) is not supported: the histogram bins x as "
        "numbers and draws a bar per bin, so type='N' would draw one bar per row "
        "instead of one per category. Drop the type (x='v')."
    )


def test_a_pairplot_diagonal_categorical_refusal_ends_in_drop_the_type():
    """pairplot's diagonal shares the same ``chains_transform=False`` row,
    on both its default histogram diagonal and a kde diagonal.
    """
    rows = pl.DataFrame(
        {"v": [float(i) for i in range(1, 21)], "w": [float(i) * 2.0 + 1.0 for i in range(1, 21)]}
    )
    with pytest.raises(ValueError) as excinfo:
        fm.pairplot(rows, vars=["v:N", "w"], diag_kind="hist").to_svg()
    assert str(excinfo.value) == (
        "pairplot(x=X('v', type='N')) is not supported: the histogram bins x as "
        "numbers and draws a bar per bin, so type='N' would draw one bar per row "
        "instead of one per category. Drop the type (x='v')."
    )
    with pytest.raises(ValueError) as excinfo:
        fm.pairplot(rows, vars=["v:N", "w"], diag_kind="kde").to_svg()
    assert str(excinfo.value) == (
        "pairplot(x=X('v', type='N')) is not supported: the kernel density estimate "
        "reads x as numbers, so type='N' would place the rows as categories. Drop "
        "the type (x='v')."
    )


#: One real categorical refusal per figure voice, each triggered by an
#: actual call rather than a hand-typed message: the typeable-remedy
#: check is only meaningful against the text the code itself raises.
_TYPEABLE_REMEDY_BUILDERS: dict[str, Callable[[], Any]] = {
    "jointplot_hist": lambda: fm.jointplot(_XY_ROWS, x="v:N", y="w").to_svg(),
    "pairplot_kde": lambda: fm.pairplot(_XY_ROWS, vars=["v:O", "w"], diag_kind="kde").to_svg(),
    "displot_hist": lambda: fm.displot(_HIST_ROWS, x="v:O", kind="hist").to_svg(),
    "regplot_fit": lambda: fm.regplot(_XY_ROWS, x=X("v", type="N"), y="w").to_svg(),
    "lmplot_fit": lambda: fm.lmplot(_XY_ROWS, x="v", y=Y("w", type="O")).to_svg(),
    "residplot_fit": lambda: fm.residplot(_XY_ROWS, x=X("v", type="N"), y="w").to_svg(),
}


@pytest.mark.parametrize("builder_id", list(_TYPEABLE_REMEDY_BUILDERS))
def test_a_figure_voiced_categorical_refusal_names_a_typeable_remedy(builder_id: str) -> None:
    """Every figure-voiced categorical refusal passes the same
    typeable-remedy check the bin=/aggregate= family's own messages do --
    including jointplot's own "Drop the type" ending, which names no
    kwarg or chained call at all. Each message comes from the real call
    that raises it, not a hand-typed string the production text could
    silently drift away from.
    """
    with pytest.raises(ValueError) as excinfo:
        _TYPEABLE_REMEDY_BUILDERS[builder_id]()
    _assert_the_remedy_is_typeable_at_the_voice(str(excinfo.value))
