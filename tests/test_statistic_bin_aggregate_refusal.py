"""A statistic refuses a ``bin=``/``aggregate=`` it cannot honor, in the caller's voice.

``mark_histogram``, ``mark_density``, ``mark_smooth`` (and
``mark_point().mark_smooth()``), ``mark_function`` with no parent data
layer, ``displot``'s hist/kde/ecdf kinds, ``residplot`` and
``regplot``/``lmplot`` each build a statistic over x/y columns it reads
raw, unfolded, or computes itself; a channel ``bin=``/``aggregate=`` the
statistic never reads would otherwise be dropped silently when the
channel is re-pointed at the statistic's output, through a keyword, an
``encode=`` entry or a shorthand string alike. Every refusal is raised as
an :class:`~ferrum._entry.EntryRefusal`, so it speaks as the public call
the caller made, never the inner mark or its own parameters, and names a
remedy that renders. The honored rows render byte-identically to a
sibling route without the option, or to a committed digest of the same
chart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y
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
    ``"y"``, ``"color"``), so the same dict drives both doors:
    ``Chart.mark_histogram(...).encode(**encode)`` and
    ``fm.displot(rows, kind="hist", **encode, **mark_kwargs)``. displot has
    no orientation of its own (vertical only) and no normalized-stack
    spelling beyond its own ``multiple='fill'`` (``position=``/channel
    ``stack=`` are ``mark_histogram``-only keywords).
    """
    cases: list[_Case] = []
    orientations = ("vertical", "horizontal") if door == "mark_histogram" else ("vertical",)
    bin_remedy = {
        "mark_histogram": "State the rule with mark_histogram's bins=, bin_count= or bin_width=, or drop bin=.",
        "displot": "State the rule with displot's bins=, or drop bin=.",
    }[door]
    density_spelling, density_remedy = {
        "mark_histogram": (
            "density=True",
            "Drop aggregate=, or draw the count with density=False.",
        ),
        "displot": ("stat='density'", "Drop aggregate=, or draw the count with stat='count'."),
    }[door]
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
                else:
                    fm.displot(rows, kind="hist", **encode, **mark_kwargs)

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
                        door,
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
            {"multiple": "fill"} if door == "displot" else {"multiple": "fill", "groupby": ["g"]},
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
    return cases


def _kde_cases() -> list[_Case]:
    """mark_density (univariate x, bivariate y) and displot(kind='kde') over x/y x bin/aggregate."""
    cases: list[_Case] = []
    reason = "the kernel density estimate reads raw rows"
    for door in ("mark_density", "displot"):
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
                        else:
                            fm.displot(_XY_ROWS, kind="kde", **encode)

                    return run

                cases.append(
                    (
                        f"{door}_kde_{channel}_{kwarg}",
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


_ALL_CASES: list[_Case] = (
    _histogram_cases("mark_histogram")
    + _histogram_cases("displot")
    + _smooth_cases()
    + _fit_cases()
    + _kde_cases()
    + _ecdf_cases()
    + _residplot_cases()
    + _mark_function_cases()
    + _spelling_cases()
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
_FIGURE_VOICES = frozenset({"displot", "residplot", "lmplot", "regplot"})
_FORBIDDEN_IN_A_FIGURE_MESSAGE = ("mark_histogram", "mark_density", "mark_smooth", "None,")
_FIGURE_CASES = [case for case in _ALL_CASES if case[2].split("(", 1)[0] in _FIGURE_VOICES]


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
class _RemedyRun:
    """One remedy, run on the door whose refusal names it.

    *refused* raises the refusal, whose message must carry *names*.
    *remedied* is the same door with the remedy applied as the message
    writes it, and *sibling* is the same door over a frame polars
    computed the same way. *without* is the remedied call with the
    remedy's own step left out; it fails because column *missing* is
    never computed.
    """

    refused: Callable[[], Any]
    names: str
    remedied: Callable[[], fm.Chart]
    sibling: Callable[[], fm.Chart]
    without: Callable[[], fm.Chart]
    without_error: type[Exception]
    missing: str


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
        without=lambda: fm.regplot(_NONSEPARABLE_ROWS, x="bin_start", y="b", method="logistic"),
        without_error=ValueError,
        missing="bin_start",
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
        without=lambda: fm.regplot(_XY_ROWS, x="bin_start", y="w", method="robust"),
        without_error=ValueError,
        missing="bin_start",
    ),
    "per_row_bin_lmplot_lm_y": _RemedyRun(
        refused=lambda: fm.lmplot(_XY_ROWS, x="v", y=Y("w", bin=True), method="lm"),
        names=_names_bin("w"),
        remedied=lambda: fm.lmplot(_XY_ROWS, x="v", y="bin_start", method="lm").transform(
            _per_row_bin("w")
        ),
        sibling=lambda: fm.lmplot(_prebinned(_XY_ROWS, "w"), x="v", y="bin_start", method="lm"),
        without=lambda: fm.lmplot(_XY_ROWS, x="v", y="bin_start", method="lm"),
        without_error=ValueError,
        missing="bin_start",
    ),
    "per_row_bin_mark_density_x": _RemedyRun(
        refused=lambda: fm.Chart(_XY_ROWS).mark_density().encode(x=X("v", bin=True)).to_svg(),
        names=_names_bin("v"),
        remedied=lambda: (
            fm.Chart(_XY_ROWS).transform(_per_row_bin("v")).mark_density().encode(x="bin_start")
        ),
        sibling=lambda: fm.Chart(_prebinned(_XY_ROWS, "v")).mark_density().encode(x="bin_start"),
        without=lambda: fm.Chart(_XY_ROWS).mark_density().encode(x="bin_start"),
        without_error=ValueError,
        missing="bin_start",
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
        without=lambda: fm.Chart(_XY_ROWS).mark_density().encode(x="v", y="bin_start"),
        without_error=ValueError,
        missing="bin_start",
    ),
    "per_row_bin_displot_kde_x": _RemedyRun(
        refused=lambda: fm.displot(_XY_ROWS, x=X("v", bin=True), kind="kde"),
        names=_names_bin("v"),
        remedied=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="kde").transform(
            _per_row_bin("v")
        ),
        sibling=lambda: fm.displot(_prebinned(_XY_ROWS, "v"), x="bin_start", kind="kde"),
        without=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="kde"),
        without_error=ValueError,
        missing="bin_start",
    ),
    "per_row_bin_displot_ecdf_x": _RemedyRun(
        refused=lambda: fm.displot(_XY_ROWS, x=X("v", bin=True), kind="ecdf"),
        names=_names_bin("v"),
        remedied=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="ecdf").transform(
            _per_row_bin("v")
        ),
        sibling=lambda: fm.displot(_prebinned(_XY_ROWS, "v"), x="bin_start", kind="ecdf"),
        without=lambda: fm.displot(_XY_ROWS, x="bin_start", kind="ecdf"),
        without_error=ValueError,
        missing="bin_start",
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
        without=lambda: fm.Chart(_XY_ROWS).mark_smooth().encode(x="v", y="bin_start"),
        without_error=ValueError,
        missing="bin_start",
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
        without=lambda: fm.residplot(_XY_ROWS, x="bin_start", y="w").transform(_per_row_bin("v")),
        without_error=_COLUMN_NOT_FOUND,
        missing="bin_start",
    ),
    "per_row_bin_residplot_y": _RemedyRun(
        refused=lambda: fm.residplot(_XY_ROWS, x="v", y=Y("w", bin=True)),
        names=_names_bin("w", " with dropna=False"),
        remedied=lambda: fm.residplot(_XY_ROWS, x="v", y="bin_start", dropna=False).transform(
            _per_row_bin("w")
        ),
        sibling=lambda: fm.residplot(_prebinned(_XY_ROWS, "w"), x="v", y="bin_start", dropna=False),
        without=lambda: fm.residplot(_XY_ROWS, x="v", y="bin_start").transform(_per_row_bin("w")),
        without_error=_COLUMN_NOT_FOUND,
        missing="bin_start",
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
        without=lambda: fm.Chart(_GROUPED_XY_ROWS).mark_histogram().encode(x="mean_v"),
        without_error=ValueError,
        missing="mean_v",
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
        without=lambda: fm.Chart(_GROUPED_XY_ROWS).mark_density().encode(x="mean_v"),
        without_error=ValueError,
        missing="mean_v",
    ),
    "transform_aggregate_displot_kde_x": _RemedyRun(
        refused=lambda: fm.displot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), kind="kde"),
        names=_names_aggregate(),
        remedied=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="kde").transform(
            _per_group_mean()
        ),
        sibling=lambda: fm.displot(_PREAGGREGATED_ROWS, x="mean_v", kind="kde"),
        without=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="kde"),
        without_error=ValueError,
        missing="mean_v",
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
        without=lambda: fm.displot(_GROUPED_XY_ROWS, x="mean_v", kind="ecdf"),
        without_error=ValueError,
        missing="mean_v",
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
        without=lambda: fm.displot(_GROUPED_HUE_ROWS, x="mean_v", kind="ecdf", hue="h"),
        without_error=ValueError,
        missing="mean_v",
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
        without=lambda: (
            fm.Chart(_GROUPED_XY_ROWS).mark_smooth(method="lm").encode(x="mean_v", y="mean_w")
        ),
        without_error=ValueError,
        missing="mean_v",
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
        without=lambda: (
            fm.Chart(_GROUPED_XY_ROWS).mark_smooth(method="lm").encode(x="mean_v", y="mean_w")
        ),
        without_error=ValueError,
        missing="mean_v",
    ),
    "transform_aggregate_lmplot_lm_x": _RemedyRun(
        refused=lambda: fm.lmplot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), y="w", method="lm"),
        names=_names_aggregate(),
        remedied=lambda: fm.lmplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="lm").transform(
            _per_group_mean()
        ),
        sibling=lambda: fm.lmplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", method="lm"),
        without=lambda: fm.lmplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="lm"),
        without_error=ValueError,
        missing="mean_v",
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
        without=lambda: fm.regplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w", method="robust"),
        without_error=ValueError,
        missing="mean_v",
    ),
    "transform_aggregate_residplot_x": _RemedyRun(
        refused=lambda: fm.residplot(_GROUPED_XY_ROWS, x=X("v", aggregate="mean"), y="w"),
        names=_names_aggregate(" with dropna=False"),
        remedied=lambda: fm.residplot(
            _GROUPED_XY_ROWS, x="mean_v", y="mean_w", dropna=False
        ).transform(_per_group_mean()),
        sibling=lambda: fm.residplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", dropna=False),
        without=lambda: fm.residplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w").transform(
            _per_group_mean()
        ),
        without_error=_COLUMN_NOT_FOUND,
        missing="mean_v",
    ),
    "transform_aggregate_residplot_y": _RemedyRun(
        refused=lambda: fm.residplot(_GROUPED_XY_ROWS, x="v", y=Y("w", aggregate="mean")),
        names=_names_aggregate(" with dropna=False"),
        remedied=lambda: fm.residplot(
            _GROUPED_XY_ROWS, x="mean_v", y="mean_w", dropna=False
        ).transform(_per_group_mean()),
        sibling=lambda: fm.residplot(_PREAGGREGATED_ROWS, x="mean_v", y="mean_w", dropna=False),
        without=lambda: fm.residplot(_GROUPED_XY_ROWS, x="mean_v", y="mean_w").transform(
            _per_group_mean()
        ),
        without_error=_COLUMN_NOT_FOUND,
        missing="mean_v",
    ),
}


@pytest.mark.parametrize("run", list(_REMEDY_RUNS.values()), ids=list(_REMEDY_RUNS))
def test_a_named_remedy_on_its_own_door_matches_the_door_over_a_precomputed_frame(
    run: _RemedyRun,
) -> None:
    """The remedy the refusal names, run on that door, draws byte-identically to
    the same door over rows polars binned or aggregated the same way; left out,
    the column it binds is never computed and the call fails.
    """
    with pytest.raises(ValueError) as excinfo:
        run.refused()
    assert run.names in str(excinfo.value)
    assert run.remedied().to_svg() == run.sibling().to_svg()
    with pytest.raises(run.without_error, match=rf"{run.missing}['\"]? not found"):
        run.without().to_svg()


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
