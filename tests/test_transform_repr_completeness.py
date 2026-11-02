"""#195 (wave 6B T11): every transform pyclass repr renders every held
constructor argument with its real value.

**Population (derived, never hand-listed).** Every class ``ferrum._core``
exports is a transform pyclass unless it is (a) on the scale protocol, i.e.
defines ``_to_scale_spec_dict`` (the hook ``ferrum.scale`` lowers through;
this covers every ``*Scale`` and ``ContinuousScheme``), or (b) one of the two
spec containers in ``_NON_TRANSFORM_CLASSES``. The rule fails closed: a new
``_core`` class of any kind lands in the population and fails the coverage leg
until it gets a table row (a transform) or an entry in
``_NON_TRANSFORM_CLASSES`` (a new non-transform family), so a new transform can
never silently escape the table. ``test_row_classes_are_real_transforms``
anchors the population to Rust's own extractor: every row class is accepted
by ``ChartSpec(transforms=[...])`` (``for_each_py_transform!``), except the
op-spec helper ``AggregateOp``, which ``Aggregate(ops=[...])`` accepts.

**Parameters (derived).** Each class's parameter list is read from its
constructor ``__text_signature__`` (``inspect.signature``), which PyO3 emits
from ``#[pyo3(signature = ...)]``. The union of a class's row kwargs must
equal that list, so a new constructor parameter without a sentinel fails.

**Rows.** ``(pyclass, kwargs with one non-default sentinel per parameter,
expected fragments)``, spec §6. A class may carry several rows where
parameters are mutually exclusive (``Bin``'s three mode spellings,
``Smooth``'s per-method knobs). Every fragment is the exact
``param=<rendered sentinel>`` text, matched at an argument boundary, never a
bare parameter name: ``"name" in repr(x)`` passes with ``name=None``
hardcoded, the bug class this test exists for (decision doc §8).

**Rulings** (decision doc §8, all six recorded here):

- ``Bin`` renders whichever of ``bins``/``bin_count``/``bin_width`` built the
  mode (the reverse projection), plus ``shared_extent``, ``groupby``,
  ``agg_field``, ``agg``, ``per_row``, ``aggregates``, ``name``.
- ``Hex``/``Raster``: ``extent_x``/``extent_y`` are facet-pinned, not
  constructor parameters; the signature-derived leg never asks for them.
- ``DensityData``/``LoessData``/``RegressionData``: constructor parameters
  only; defaulted struct fields with no parameter are not held arguments.
- ``JoinAggregate`` renders its one parameter, ``name``.
- ``Aggregate`` renders ``ops=[AggregateOp(...), ...]``, the held values.
- ``LetterValue``: ``k_proportion`` is rendered inside ``k_depth``'s
  ``Proportion`` arm (``_RENDERED_UNDER``); its sentinel is asserted there.

``AggregateOp`` renders ``fn_=`` (the parameter's spelling) with the
canonical ``AggFn`` spelling, so the ``var`` alias renders ``variance``.

**Vacuity control.** Every fragment of a defaulted parameter must be absent
from the repr of the same class built from its required arguments alone, so
a sentinel equal to the default (or a hardcoded rendering) cannot pass.

Round-trip (``eval(repr(x)) == x``) is follow-up F7, not this rule; Rust
``{:?}`` renderings such as ``Fixed { value: 0.37 }`` are the real value.
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple

import pytest

from ferrum import _core
from tests._repr_probe import has_arg

# Non-transform classes the scale-protocol rule does not already exclude.
_NON_TRANSFORM_CLASSES = {
    "ChartSpec": "the chart spec container",
    "EncodingSpec": "the encoding spec container",
}

# (class, parameter) -> the argument name whose rendering carries its value.
_RENDERED_UNDER = {("LetterValue", "k_proportion"): "k_depth"}


class Row(NamedTuple):
    cls: str
    kwargs: dict[str, Any]
    expected: dict[str, str]  # parameter -> exact rendered fragment


def _op(field: str, fn: str, as_: str) -> Any:
    return _core.AggregateOp(field, fn, as_)


ROWS: list[Row] = [
    # --- Bin: one row per mode spelling (ruling 1) ---
    Row(
        "Bin",
        dict(
            field="bin_f",
            bins="fd",
            extent=(-3.25, 7.5),
            nice=False,
            cumulative=True,
            shared_extent=True,
            groupby=["bin_g"],
            agg_field="bin_w",
            agg="median",
            name="bin_nm",
        ),
        dict(
            field="field='bin_f'",
            bins="bins='freedman_diaconis'",
            extent="extent=Some((-3.25, 7.5))",
            nice="nice=False",
            cumulative="cumulative=True",
            shared_extent="shared_extent=True",
            groupby='groupby=["bin_g"]',
            agg_field='agg_field=Some("bin_w")',
            agg='agg=Some("median")',
            name='name=Some("bin_nm")',
        ),
    ),
    Row(
        "Bin",
        dict(field="bin_f", bin_count=17),
        dict(field="field='bin_f'", bin_count="bin_count=Some(17)"),
    ),
    Row(
        "Bin",
        dict(field="bin_f", bin_width=2.75),
        dict(field="field='bin_f'", bin_width="bin_width=Some(2.75)"),
    ),
    Row(
        "Bin",
        dict(field="bin_f", per_row=True),
        dict(field="field='bin_f'", per_row="per_row=True"),
    ),
    Row(
        "Bin",
        dict(
            field="bin_f",
            aggregates=[
                _core.AggregateOp("bin_w", "mean", "bin_m"),
                _core.AggregateOp("bin_w", "var", "bin_v"),
            ],
        ),
        dict(
            field="field='bin_f'",
            aggregates=(
                "aggregates=[AggregateOp(field='bin_w', fn_='mean', as_='bin_m'), "
                "AggregateOp(field='bin_w', fn_='variance', as_='bin_v')]"
            ),
        ),
    ),
    Row(
        "Bin2D",
        dict(
            x="b2_x",
            y="b2_y",
            bins_x=13,
            bins_y=2.5,
            extent_x=(-1.5, 4.25),
            extent_y=(-2.5, 8.75),
            cumulative=True,
            name="b2_nm",
        ),
        dict(
            x="x='b2_x'",
            y="y='b2_y'",
            bins_x="bins_x=Fixed { n: 13 }",
            bins_y="bins_y=Width { w: 2.5 }",
            extent_x="extent_x=Some((-1.5, 4.25))",
            extent_y="extent_y=Some((-2.5, 8.75))",
            cumulative="cumulative=true",
            name='name=Some("b2_nm")',
        ),
    ),
    Row(
        "Kde",
        dict(
            field="kde_f",
            bandwidth=0.37,
            bw_adjust=1.7,
            n=333,
            extent=(-4.5, 6.5),
            cumulative=True,
            shared_extent=True,
            kernel="epanechnikov",
            groupby=["kde_g"],
            count_scale=23,
            name="kde_nm",
        ),
        dict(
            field="field='kde_f'",
            bandwidth="bandwidth=Fixed { value: 0.37 }",
            bw_adjust="bw_adjust=1.7",
            n="n=333",
            extent="extent=Some((-4.5, 6.5))",
            cumulative="cumulative=True",
            shared_extent="shared_extent=True",
            kernel="kernel='epanechnikov'",
            groupby='groupby=["kde_g"]',
            count_scale="count_scale=Some(Fixed { n: 23 })",
            name='name=Some("kde_nm")',
        ),
    ),
    # --- Smooth: per-method knobs are mutually exclusive ---
    Row(
        "Smooth",
        dict(
            x="sm_x",
            y="sm_y",
            method="lm",
            degree=3,
            ci=0.83,
            n=77,
            seed=91,
            x_bins=9,
            x_estimator="median",
            output="residuals",
            inject_zero_ref=True,
            inject_metrics=True,
            groupby=["sm_g"],
            name="sm_nm",
            x_range=(-1.25, 3.5),
            x_range_pad=0.08,
        ),
        dict(
            x="x='sm_x'",
            y="y='sm_y'",
            method="method=Poly { degree: 3 }",
            degree="degree=3",
            ci="ci=Some(0.83)",
            n="n=77",
            seed="seed=91",
            x_bins="x_bins=Some(9)",
            x_estimator='x_estimator=Some("median")',
            output="output=Residuals",
            inject_zero_ref="inject_zero_ref=true",
            inject_metrics="inject_metrics=true",
            groupby='groupby=["sm_g"]',
            name='name=Some("sm_nm")',
            x_range="x_range=Some([-1.25, 3.5])",
            x_range_pad="x_range_pad=Some(0.08)",
        ),
    ),
    Row(
        "Smooth",
        dict(x="sm_x", y="sm_y", bandwidth=0.41),
        dict(x="x='sm_x'", y="y='sm_y'", bandwidth="bandwidth=0.41"),
    ),
    Row(
        "Smooth",
        dict(x="sm_x", y="sm_y", method="gam", k=7),
        dict(x="x='sm_x'", y="y='sm_y'", method="method=Gam", k="k=7"),
    ),
    # --- Aggregate renders the held ops (ruling 5) ---
    Row(
        "Aggregate",
        dict(
            ops=[_op("op_v", "stdev", "op_as"), _op("op_w", "var", "op_as2")],
            groupby=["ag_g"],
            name="ag_nm",
        ),
        dict(
            ops=(
                "ops=[AggregateOp(field='op_v', fn_='stdev', as_='op_as'), "
                "AggregateOp(field='op_w', fn_='variance', as_='op_as2')]"
            ),
            groupby='groupby=["ag_g"]',
            name='name=Some("ag_nm")',
        ),
    ),
    Row(
        "AggregateOp",
        dict(field="op_v", fn_="stdev", as_="op_as"),
        dict(field="field='op_v'", fn_="fn_='stdev'", as_="as_='op_as'"),
    ),
    Row(
        "AggregateOp",
        dict(field="op_v", fn_="var", as_="op_as"),
        dict(field="field='op_v'", fn_="fn_='variance'", as_="as_='op_as'"),
    ),
    Row(
        "Summary",
        dict(
            field="su_f",
            groupby=["su_g"],
            error_fn="stdev",
            ci=0.83,
            n_boot=321,
            seed=91,
            name="su_nm",
        ),
        dict(
            field="field='su_f'",
            groupby='groupby=["su_g"]',
            error_fn="error_fn=Stdev",
            ci="ci=0.83",
            n_boot="n_boot=321",
            seed="seed=91",
            name='name=Some("su_nm")',
        ),
    ),
    Row(
        "Outliers",
        dict(field="ou_f", groupby=["ou_g"], extent=2.25, name="ou_nm"),
        dict(
            field="field='ou_f'",
            groupby='groupby=["ou_g"]',
            extent="extent=2.25",
            name='name=Some("ou_nm")',
        ),
    ),
    Row(
        "ErrorExtent",
        dict(
            field="ee_f",
            method="iqr",
            groupby=["ee_g"],
            seed=91,
            n_boot=321,
            name="ee_nm",
        ),
        dict(
            field="field='ee_f'",
            method="method=Iqr",
            groupby='groupby=["ee_g"]',
            seed="seed=91",
            n_boot="n_boot=321",
            name='name=Some("ee_nm")',
        ),
    ),
    Row(
        "BoxStats",
        dict(field="bx_f", groupby=["bx_g"], whisker_extent=2.25, name="bx_nm"),
        dict(
            field="field='bx_f'",
            groupby='groupby=["bx_g"]',
            whisker_extent="whisker_extent=IqrMultiplier(2.25)",
            name='name=Some("bx_nm")',
        ),
    ),
    Row(
        "Violin",
        dict(
            field="vi_f",
            groupby=["vi_g"],
            bandwidth=0.37,
            bw_adjust=1.7,
            n=333,
            width=0.61,
            shared_extent=True,
            horizontal=True,
            name="vi_nm",
        ),
        dict(
            field="field='vi_f'",
            groupby='groupby=["vi_g"]',
            bandwidth="bandwidth=Fixed { value: 0.37 }",
            bw_adjust="bw_adjust=1.7",
            n="n=333",
            width="width=0.61",
            shared_extent="shared_extent=True",
            horizontal="horizontal=True",
            name='name=Some("vi_nm")',
        ),
    ),
    Row(
        "Kde2D",
        dict(
            x="k2_x",
            y="k2_y",
            bandwidth=0.37,
            n=77,
            extent=(-4.5, 6.5, -3.5, 5.5),
            groupby=["k2_g"],
            name="k2_nm",
        ),
        dict(
            x="x='k2_x'",
            y="y='k2_y'",
            bandwidth="bandwidth=Fixed { value: 0.37 }",
            n="n=77",
            extent="extent=Some((-4.5, 6.5, -3.5, 5.5))",
            groupby='groupby=["k2_g"]',
            name='name=Some("k2_nm")',
        ),
    ),
    Row(
        "Contour",
        dict(thresholds=11, fill=True, smooth=False, name="co_nm"),
        dict(
            thresholds="thresholds=11",
            fill="fill=true",
            smooth="smooth=false",
            name='name=Some("co_nm")',
        ),
    ),
    Row(
        "QQ",
        dict(
            field="qq_f",
            distribution="uniform",
            dequantize=True,
            emit_line=False,
            seed=91,
            groupby=["qq_g"],
            name="qq_nm",
        ),
        dict(
            field="field='qq_f'",
            distribution="distribution='uniform'",
            dequantize="dequantize=true",
            emit_line="emit_line=false",
            seed="seed=91",
            groupby='groupby=["qq_g"]',
            name='name=Some("qq_nm")',
        ),
    ),
    Row(
        "Linkage",
        dict(
            method="average",
            metric="cosine",
            axis="columns",
            z_score="rows",
            standard_scale="columns",
            name="li_nm",
        ),
        dict(
            method="method=Average",
            metric="metric=Cosine",
            axis="axis=Columns",
            z_score="z_score=Some(Rows)",
            standard_scale="standard_scale=Some(Columns)",
            name='name=Some("li_nm")',
        ),
    ),
    Row(
        "Raster",
        dict(
            x="ra_x",
            y="ra_y",
            aggregate="mean",
            field="ra_f",
            resolution=37,
            min_count=4,
            log_scale=True,
            name="ra_nm",
        ),
        dict(
            x="x='ra_x'",
            y="y='ra_y'",
            aggregate="aggregate='mean'",
            field='field=Some("ra_f")',
            resolution="resolution=Fixed(37)",
            min_count="min_count=Some(4)",
            log_scale="log_scale=true",
            name='name=Some("ra_nm")',
        ),
    ),
    Row(
        "Hex",
        dict(
            x="hx_x",
            y="hx_y",
            bin_size=0.83,
            aggregate="max",
            field="hx_f",
            name="hx_nm",
        ),
        dict(
            x="x='hx_x'",
            y="y='hx_y'",
            bin_size="bin_size=Some(0.83)",
            aggregate="aggregate='max'",
            field='field=Some("hx_f")',
            name='name=Some("hx_nm")',
        ),
    ),
    Row(
        "Swarm",
        dict(
            category="sw_c",
            value="sw_v",
            point_size=7.5,
            spacing=1.25,
            side="left",
            orient="horizontal",
            dodge="sw_d",
            name="sw_nm",
        ),
        dict(
            category="category='sw_c'",
            value="value='sw_v'",
            point_size="point_size=7.5",
            spacing="spacing=1.25",
            side="side=Left",
            orient="orient=Horizontal",
            dodge='dodge=Some("sw_d")',
            name='name=Some("sw_nm")',
        ),
    ),
    Row(
        "Unpivot",
        dict(
            id_vars=["up_id"],
            value_vars=["up_v"],
            var_name="up_var",
            value_name="up_val",
            name="up_nm",
        ),
        dict(
            id_vars='id_vars=["up_id"]',
            value_vars='value_vars=Some(["up_v"])',
            var_name="var_name='up_var'",
            value_name="value_name='up_val'",
            name='name=Some("up_nm")',
        ),
    ),
    Row(
        "Reorder",
        dict(by="ro_by", drop_index=False, from_="ro_from", axis="columns", name="ro_nm"),
        dict(
            by="by='ro_by'",
            drop_index="drop_index=false",
            # The parameter's spelling, as AggregateOp's `fn_=` (ruling §8).
            from_='from_=Some("ro_from")',
            axis="axis='columns'",
            name='name=Some("ro_nm")',
        ),
    ),
    Row(
        "ReferenceLine",
        dict(
            x_field="rl_x",
            y_field="rl_y",
            x=(-1.5, 2.25),
            y=(-3.5, 4.75),
            name="rl_nm",
        ),
        dict(
            x_field="x_field='rl_x'",
            y_field="y_field='rl_y'",
            x="x=(-1.5, 2.25)",
            y="y=(-3.5, 4.75)",
            name='name=Some("rl_nm")',
        ),
    ),
    # --- LetterValue: k_proportion rendered inside k_depth (ruling 6) ---
    Row(
        "LetterValue",
        dict(
            value="lv_v",
            group="lv_g",
            groupby=["lv_h"],
            k_depth="tukey",
            outlier_threshold=2.25,
            name="lv_nm",
            sort={"op": "mean", "order": "descending"},
        ),
        dict(
            value="value='lv_v'",
            group='group=Some("lv_g")',
            groupby='groupby=["lv_h"]',
            k_depth="k_depth=Tukey",
            outlier_threshold="outlier_threshold=2.25",
            name='name=Some("lv_nm")',
            sort="sort=Some(LetterValueSort { op: Mean, order: Desc })",
        ),
    ),
    Row(
        "LetterValue",
        dict(value="lv_v", k_proportion=0.023),
        dict(value="value='lv_v'", k_proportion="k_depth=Proportion { p: 0.023 }"),
    ),
    Row(
        "Logistic",
        dict(
            x="lo_x",
            y="lo_y",
            n_grid=77,
            ci=0.83,
            max_iter=37,
            tol=3e-5,
            groupby=["lo_g"],
            name="lo_nm",
        ),
        dict(
            x="x='lo_x'",
            y="y='lo_y'",
            n_grid="n_grid=77",
            ci="ci=Some(0.83)",
            max_iter="max_iter=37",
            tol="tol=0.00003",
            groupby='groupby=["lo_g"]',
            name='name=Some("lo_nm")',
        ),
    ),
    Row(
        "Glm",
        dict(
            x="gl_x",
            y="gl_y",
            family="poisson",
            link="log",
            n_grid=77,
            ci=0.83,
            max_iter=37,
            tol=3e-5,
            groupby=["gl_g"],
            name="gl_nm",
        ),
        dict(
            x="x='gl_x'",
            y="y='gl_y'",
            family="family='poisson'",
            link="link='log'",
            n_grid="n_grid=77",
            ci="ci=Some(0.83)",
            max_iter="max_iter=37",
            tol="tol=0.00003",
            groupby='groupby=["gl_g"]',
            name='name=Some("gl_nm")',
        ),
    ),
    Row(
        "Robust",
        dict(
            x="rb_x",
            y="rb_y",
            n_grid=77,
            ci=0.83,
            huber_c=2.25,
            max_iter=37,
            tol=3e-5,
            output="residuals",
            inject_zero_ref=True,
            inject_metrics=True,
            groupby=["rb_g"],
            name="rb_nm",
            x_range=(-1.25, 3.5),
            x_range_pad=0.08,
        ),
        dict(
            x="x='rb_x'",
            y="y='rb_y'",
            n_grid="n_grid=77",
            ci="ci=Some(0.83)",
            huber_c="huber_c=2.25",
            max_iter="max_iter=37",
            tol="tol=0.00003",
            output="output=Residuals",
            inject_zero_ref="inject_zero_ref=true",
            inject_metrics="inject_metrics=true",
            groupby='groupby=["rb_g"]',
            name='name=Some("rb_nm")',
            x_range="x_range=Some([-1.25, 3.5])",
            x_range_pad="x_range_pad=Some(0.08)",
        ),
    ),
    Row("PyIdentity", dict(name="id_nm"), dict(name='name=Some("id_nm")')),
    # --- JoinAggregate renders its one parameter (ruling 4) ---
    Row("JoinAggregate", dict(name="ja_nm"), dict(name='name=Some("ja_nm")')),
    # --- Data* classes: constructor parameters only (ruling 3) ---
    Row(
        "DensityData",
        dict(field="dd_f", name="dd_nm"),
        dict(field="field='dd_f'", name='name=Some("dd_nm")'),
    ),
    Row(
        "RegressionData",
        dict(x="rd_x", y="rd_y", method="poly", order=3, name="rd_nm"),
        dict(
            x="x='rd_x'",
            y="y='rd_y'",
            method="method='poly'",
            order="order=3",
            name='name=Some("rd_nm")',
        ),
    ),
    Row(
        "LoessData",
        dict(x="ld_x", y="ld_y", bandwidth=0.41, name="ld_nm"),
        dict(
            x="x='ld_x'",
            y="y='ld_y'",
            bandwidth="bandwidth=0.41",
            name='name=Some("ld_nm")',
        ),
    ),
    Row(
        "Impute",
        dict(field="im_f", value=2.75, name="im_nm"),
        dict(
            field="field='im_f'",
            value="value=Some(2.75)",
            name='name=Some("im_nm")',
        ),
    ),
    Row(
        "Impute",
        dict(field="im_f", method="mean"),
        dict(field="field='im_f'", method="method='mean'"),
    ),
    Row(
        "Flatten",
        dict(fields=["fl_a", "fl_b"], as_=["fl_as_a", "fl_as_b"], name="fl_nm"),
        dict(
            fields='fields=["fl_a", "fl_b"]',
            as_='as_=Some(["fl_as_a", "fl_as_b"])',
            name='name=Some("fl_nm")',
        ),
    ),
    Row(
        "Sample",
        dict(n=37, seed=91, name="sa_nm"),
        dict(n="n=37", seed="seed=91", name='name=Some("sa_nm")'),
    ),
    Row(
        "TopK",
        dict(
            n=37,
            field="tk_f",
            op="max",
            sort="ascending",
            groupby=["tk_g"],
            name="tk_nm",
        ),
        dict(
            n="n=37",
            field="field='tk_f'",
            op="op='max'",
            sort="sort='ascending'",
            groupby='groupby=Some(["tk_g"])',
            name='name=Some("tk_nm")',
        ),
    ),
    Row(
        "TimeUnit",
        dict(field="tu_f", unit="month", utc=True, as_="tu_as", name="tu_nm"),
        dict(
            field="field='tu_f'",
            unit="unit='month'",
            utc="utc=True",
            as_='as_=Some("tu_as")',
            name='name=Some("tu_nm")',
        ),
    ),
]


def _discovered_transform_classes() -> dict[str, type]:
    return {
        name: obj
        for name, obj in vars(_core).items()
        if inspect.isclass(obj)
        and not hasattr(obj, "_to_scale_spec_dict")
        and name not in _NON_TRANSFORM_CLASSES
    }


def _params(cls_name: str) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(getattr(_core, cls_name)).parameters)


def _build(row: Row) -> Any:
    return getattr(_core, row.cls)(**row.kwargs)


def _build_required_only(row: Row) -> Any:
    required = {
        p: row.kwargs[p]
        for p, spec in _params(row.cls).items()
        if spec.default is inspect.Parameter.empty
    }
    return getattr(_core, row.cls)(**required)


def _row_id(row: Row) -> str:
    return f"{row.cls}[{','.join(row.kwargs)}]"


# --- coverage legs --------------------------------------------------------


def test_every_discovered_transform_class_has_a_row() -> None:
    population = set(_discovered_transform_classes())
    rowed = {row.cls for row in ROWS}
    assert population - rowed == set(), "transform pyclass(es) without a table row"
    assert rowed - population == set(), "table row(s) for a class _core no longer exports"


def test_population_is_not_vacuous() -> None:
    # 34 for_each_py_transform! entries plus the AggregateOp op-spec class.
    assert len(_discovered_transform_classes()) >= 35


@pytest.mark.parametrize("cls_name", sorted({row.cls for row in ROWS}))
def test_rows_cover_every_constructor_parameter(cls_name: str) -> None:
    covered = {p for row in ROWS if row.cls == cls_name for p in row.kwargs}
    assert covered == set(_params(cls_name))


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_row_expects_one_fragment_per_kwarg(row: Row) -> None:
    assert set(row.expected) == set(row.kwargs)
    for param, frag in row.expected.items():
        owner = _RENDERED_UNDER.get((row.cls, param), param)
        assert frag.startswith(f"{owner}="), (param, frag)


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_row_classes_are_real_transforms(row: Row) -> None:
    obj = _build(row)
    if row.cls == "AggregateOp":
        _core.Aggregate([obj])
    else:
        _core.ChartSpec(mark="point", transforms=[obj])


# --- the rule -------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_repr_renders_every_held_argument(row: Row) -> None:
    text = repr(_build(row))
    missing = {param: frag for param, frag in row.expected.items() if not has_arg(text, frag)}
    assert missing == {}, f"{text}\n missing: {missing}"


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_sentinels_are_not_the_default_rendering(row: Row) -> None:
    """Vacuity control: a defaulted parameter's fragment must be absent from
    the required-arguments-only repr, and the two reprs must differ."""
    params = _params(row.cls)
    defaulted = [p for p in row.kwargs if params[p].default is not inspect.Parameter.empty]
    if not defaulted:
        pytest.skip("row holds required parameters only")
    default_text = repr(_build_required_only(row))
    assert repr(_build(row)) != default_text
    for param in defaulted:
        assert not has_arg(default_text, row.expected[param]), (
            param,
            default_text,
        )
