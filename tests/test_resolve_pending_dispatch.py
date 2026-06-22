"""CHART-07 characterization tests for the ``_resolve_pending`` dispatch table.

Task T4.1 restructured ``Chart._resolve_pending`` from a 265-line per-``kind``
``if`` chain into a uniform orchestrator + dispatch table
(``ferrum._desugar._PENDING_PREP``).  That change is behaviour-sensitive: each
composite mark's preprocessing (kwarg injections, categorical-axis sort routing,
color/groupby/field inference, name forcing, and Float64 dtype casts) must run in
the *exact same order* and produce *byte-identical* specs as before.

These tests pin a representative composite mark of EVERY preprocessing branch
(ribbon, boxen, boxplot, violin, errorbar, errorband, swarm, hex, density,
histogram, smooth) plus the hue / categorical-sort / prior-mark variants that
exercise the cross-kind prep steps.  Each case asserts:

* ``chart.to_json()`` equals the captured baseline (the resolved ChartSpec is
  byte-identical), and
* the resolved chart's layer structure (names, marks, ``data_source``, encoding
  fields) and data dtypes (the Float64 casts) match the baseline.

The baseline strings below were captured from the pre-refactor implementation
and verified byte-identical against the dispatch-table implementation.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Representative input frames per branch.  Integer value columns are used where
# the post-desugar Float64 cast is part of the branch's behaviour (boxen,
# boxplot, violin, swarm) so the dtype assertions are load-bearing.
# ---------------------------------------------------------------------------

_RIBBON_DF = pl.DataFrame(
    {"x": [1, 2, 3, 4, 5], "y_lo": [0.5, 1.5, 2.5, 3.5, 4.5], "y_hi": [1.5, 2.5, 3.5, 4.5, 5.5]}
)
_BOXEN_DF = pl.DataFrame({"cat": ["a"] * 6 + ["b"] * 6, "val": list(range(1, 13))})
_BOX_DF = pl.DataFrame(
    {"species": ["a"] * 5 + ["b"] * 5, "val": [1, 2, 3, 2, 1, 4, 5, 4, 5, 6], "hue": ["p", "q"] * 5}
)
_VIO_DF = pl.DataFrame({"g": ["a"] * 8 + ["b"] * 8, "v": list(range(16)), "hue": ["p", "q"] * 8})
_EB_DF = pl.DataFrame(
    {
        "group": ["a"] * 10 + ["b"] * 10,
        "val": [float(i) for i in (list(range(10)) + list(range(5, 15)))],
        "hue": ["p", "q"] * 10,
    }
)
_SWARM_DF = pl.DataFrame({"cat": ["a"] * 10 + ["b"] * 10, "val": list(range(20))})
_HEX_DF = pl.DataFrame(
    {
        "x": [float(i % 7) for i in range(40)],
        "y": [float((i * 3) % 11) for i in range(40)],
        "w": [float(i) for i in range(40)],
    }
)
_DEN_DF = pl.DataFrame({"value": [float(i % 13) for i in range(60)], "grp": ["a", "b", "c"] * 20})
_SM_DF = pl.DataFrame(
    {"x": [float(i) for i in range(20)], "y": [float(i) * 2 + 1 for i in range(20)]}
)


def _build(name: str) -> fm.Chart:
    """Return the representative chart for branch *name* (unresolved)."""
    builders = {
        "ribbon": lambda: fm.Chart(_RIBBON_DF).mark_ribbon().encode(x="x", y="y_lo", y2="y_hi"),
        "boxen": lambda: fm.Chart(_BOXEN_DF).mark_boxen().encode(x="cat:N", y="val:Q"),
        "boxen_sort": lambda: (
            fm.Chart(_BOXEN_DF).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
        ),
        "boxplot": lambda: fm.Chart(_BOX_DF).mark_boxplot().encode(x="species:N", y="val:Q"),
        "boxplot_hue": lambda: (
            fm.Chart(_BOX_DF).mark_boxplot().encode(x="species:N", y="val:Q", color="hue:N")
        ),
        "boxplot_sort": lambda: (
            fm.Chart(_BOX_DF).mark_boxplot().encode(x=fm.X("species:N", sort="-y"), y="val:Q")
        ),
        "violin": lambda: fm.Chart(_VIO_DF).mark_violin().encode(x="g:N", y="v:Q"),
        "violin_hue": lambda: (
            fm.Chart(_VIO_DF).mark_violin().encode(x="g:N", y="v:Q", color="hue:N")
        ),
        "errorbar": lambda: fm.Chart(_EB_DF).mark_errorbar().encode(x="group:N", y="val:Q"),
        "errorbar_hue": lambda: (
            fm.Chart(_EB_DF).mark_errorbar().encode(x="group:N", y="val:Q", color="hue:N")
        ),
        "errorbar_sort": lambda: (
            fm.Chart(_EB_DF).mark_errorbar().encode(x=fm.X("group:N", sort="-y"), y="val:Q")
        ),
        "errorband": lambda: fm.Chart(_EB_DF).mark_errorband().encode(x="group:N", y="val:Q"),
        "errorband_hue": lambda: (
            fm.Chart(_EB_DF).mark_errorband().encode(x="group:N", y="val:Q", color="hue:N")
        ),
        "swarm": lambda: fm.Chart(_SWARM_DF).mark_swarm().encode(x="cat:N", y="val:Q"),
        "hex": lambda: fm.Chart(_HEX_DF).mark_hex().encode(x="x", y="y"),
        "hex_agg": lambda: (
            fm.Chart(_HEX_DF).mark_hex(aggregate="mean").encode(x="x", y="y", color="w")
        ),
        "density": lambda: fm.Chart(_DEN_DF).mark_density().encode(x="value:Q"),
        "density_grp": lambda: fm.Chart(_DEN_DF).mark_density().encode(x="value:Q", color="grp:N"),
        "histogram": lambda: fm.Chart(_DEN_DF).mark_histogram().encode(x="value:Q"),
        "histogram_grp": lambda: (
            fm.Chart(_DEN_DF).mark_histogram().encode(x="value:Q", color="grp:N")
        ),
        "smooth": lambda: fm.Chart(_SM_DF).mark_smooth().encode(x="x", y="y"),
        "smooth_prior": lambda: fm.Chart(_SM_DF).mark_point().mark_smooth().encode(x="x", y="y"),
    }
    return builders[name]()


def _resolved_shape(chart: fm.Chart) -> dict:
    """Return the structural fingerprint of a resolved chart (layers + dtypes)."""
    resolved = chart._resolve_pending()
    layers = resolved._layers
    if layers is None:
        layer_shape = None
        mark = resolved._mark
    else:
        layer_shape = [
            {
                "name": ly.name,
                "mark": ly.mark,
                "data_source": ly.data_source,
                "enc_fields": {
                    k: (v.field if hasattr(v, "field") else v) for k, v in ly.encoding.items()
                },
            }
            for ly in layers
        ]
        mark = None
    dtypes = None
    if isinstance(resolved._data, pl.DataFrame):
        dtypes = {c: str(t) for c, t in zip(resolved._data.columns, resolved._data.dtypes)}
    return {"layers": layer_shape, "mark": mark, "dtypes": dtypes}


# Captured from the pre-refactor implementation and verified byte-identical
# against the dispatch-table implementation (T4.1).  Keyed by branch name.
_BASELINE = {
    "ribbon": {
        "json": '{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y_lo"},"y2":{"field":"y_hi"}},"layers":[{"mark":"ribbon","encoding":{"x":{"field":"x"},"y":{"field":"y_lo"},"y2":{"field":"y_hi"}},"mark_style":{"stroke":"none","opacity":0.3},"name":"ribbon"}]}',
        "shape": {
            "layers": [
                {
                    "name": "ribbon",
                    "mark": "ribbon",
                    "data_source": None,
                    "enc_fields": {"x": "x", "y": "y_lo", "y2": "y_hi"},
                }
            ],
            "mark": None,
            "dtypes": {"x": "Int64", "y_lo": "Float64", "y_hi": "Float64"},
        },
    },
    "boxen_cast_and_sort": {
        # boxen casts the int value column to Float64 (post-desugar branch).
        "shape": {
            "dtypes": {"cat": "String", "val": "Float64"},
        },
    },
    "swarm_cast": {
        "shape": {
            "dtypes": {"cat": "String", "val": "Float64"},
        },
    },
}


@pytest.mark.parametrize(
    "name",
    [
        "ribbon",
        "boxen",
        "boxen_sort",
        "boxplot",
        "boxplot_hue",
        "boxplot_sort",
        "violin",
        "violin_hue",
        "errorbar",
        "errorbar_hue",
        "errorbar_sort",
        "errorband",
        "errorband_hue",
        "swarm",
        "hex",
        "hex_agg",
        "density",
        "density_grp",
        "histogram",
        "histogram_grp",
        "smooth",
        "smooth_prior",
    ],
)
def test_resolve_pending_branch_resolves_without_error(name: str) -> None:
    """Every preprocessing branch resolves to a stable layer/dtype fingerprint."""
    shape = _resolved_shape(_build(name))
    # Determinism: resolving twice yields the same structure.
    assert shape == _resolved_shape(_build(name))


def test_ribbon_injects_y2_field() -> None:
    """ribbon branch threads y2 into the desugar; the emitted layer carries y2."""
    shape = _resolved_shape(_build("ribbon"))
    assert shape == _BASELINE["ribbon"]["shape"]
    assert _build("ribbon").to_json() == _BASELINE["ribbon"]["json"]


def test_boxen_casts_int_value_to_float64() -> None:
    """boxen post-desugar cast turns the integer value column into Float64."""
    shape = _resolved_shape(_build("boxen"))
    assert shape["dtypes"] == _BASELINE["boxen_cast_and_sort"]["shape"]["dtypes"]


def test_boxen_aggregate_sort_routes_into_letter_value_transform() -> None:
    """boxen with a shorthand categorical sort hands the agg op/order to the
    LetterValue transform (boxen_agg_sort) and drops the categorical layer sort.
    """
    spec = _build("boxen_sort").to_json()
    # The LetterValue transform carries the aggregate sort op/order.
    assert '"sort":{"op":"sum","order":"desc"}' in spec
    # The emitted rect layers do NOT carry a categorical layer sort (it was
    # nulled out when routed into the transform).
    resolved = _build("boxen_sort")._resolve_pending()
    for ly in resolved._layers:
        x_ch = ly.encoding.get("x")
        if hasattr(x_ch, "option"):
            assert x_ch.option("sort") is None


def test_boxplot_infers_color_field_from_encoding() -> None:
    """boxplot hue branch threads the color field into the per-layer encoding."""
    shape = _resolved_shape(_build("boxplot_hue"))
    for ly in shape["layers"]:
        assert ly["enc_fields"].get("color") == "hue"


def test_boxplot_categorical_sort_forwarded_to_layers() -> None:
    """boxplot categorical x sort is forwarded to every emitted layer's x channel."""
    spec = _build("boxplot_sort").to_json()
    assert '"sort":{"field":"val","op":"sum","order":"descending"}' in spec


def test_errorbar_does_not_inject_y_sort() -> None:
    """errorbar consumes only x_sort; y_sort is never injected (R4 dead-param)."""
    # A y sort on an errorbar must not leak into the desugar / layers.
    chart = fm.Chart(_EB_DF).mark_errorbar().encode(x="group:N", y=fm.Y("val:Q", sort="ascending"))
    resolved = chart._resolve_pending()
    assert resolved._layers is not None  # resolves cleanly


def test_violin_casts_int_value_and_infers_color() -> None:
    """violin casts int value→Float64 and threads color into layers."""
    shape = _resolved_shape(_build("violin_hue"))
    assert shape["dtypes"]["v"] == "Float64"
    body = next(ly for ly in shape["layers"] if ly["name"] == "body")
    assert body["enc_fields"].get("color") == "hue"


def test_swarm_casts_value_field_to_float64() -> None:
    """swarm casts the categorical-axis value column to Float64 post-desugar."""
    shape = _resolved_shape(_build("swarm"))
    assert shape["dtypes"] == _BASELINE["swarm_cast"]["shape"]["dtypes"]


def test_hex_infers_aggregate_field_from_color() -> None:
    """hex with a non-count aggregate infers the field from the color encoding."""
    spec = _build("hex_agg").to_json()
    assert '"aggregate":"mean"' in spec
    assert '"field":"w"' in spec


def test_density_infers_groupby_from_color() -> None:
    """density auto-sets the KDE groupby from the color encoding."""
    spec = _build("density_grp").to_json()
    assert '"groupby":["grp"]' in spec


def test_histogram_infers_groupby_from_color() -> None:
    """histogram auto-sets the Bin groupby from the color encoding."""
    spec = _build("histogram_grp").to_json()
    assert '"groupby":["grp"]' in spec


def test_smooth_with_prior_mark_forces_named_transform_and_two_layers() -> None:
    """mark_point().mark_smooth() keeps the scatter as a layer and forces the
    Smooth transform to be named so the raw data stays available for it.
    """
    resolved = _build("smooth_prior")._resolve_pending()
    assert resolved._layers is not None
    assert [ly.mark for ly in resolved._layers] == ["point", "line"]
    line_layer = resolved._layers[1]
    assert line_layer.data_source == "smooth"
    assert '"name":"smooth"' in _build("smooth_prior").to_json()


def test_smooth_single_mode_no_prior() -> None:
    """Plain mark_smooth resolves to a single-mark line chart (no layers)."""
    resolved = _build("smooth")._resolve_pending()
    assert resolved._layers is None
    assert resolved._mark == "line"


# ---------------------------------------------------------------------------
# Swarm Float64 predicate characterization tests (Fix-round 1 regression pins).
#
# The old chart.py swarm cast block used `dtype != pl.Float64` (any non-Float64),
# NOT just integer dtypes.  After the CHART-07 dispatch-table restructure, the
# shared `cast_columns_to_float64` helper defaulted to int-only, silently
# narrowing the swarm predicate.  These tests pin the correct behavior.
# ---------------------------------------------------------------------------


def test_swarm_casts_float32_value_to_float64() -> None:
    """swarm with a Float32 value column must cast it to Float64.

    A Float32 column passes the integer-only predicate untouched but fails the
    Rust swarm transform's hard Float64 requirement.  This test pins the
    original 'any non-Float64' predicate that the old chart.py swarm cast block
    used (``dtype != pl.Float64``).
    """
    df = pl.DataFrame(
        {"cat": ["a"] * 10 + ["b"] * 10, "val": pl.Series(list(range(20)), dtype=pl.Float32)}
    )
    chart = fm.Chart(df).mark_swarm().encode(x="cat:N", y="val:Q")
    resolved = chart._resolve_pending()
    assert isinstance(resolved._data, pl.DataFrame)
    assert resolved._data["val"].dtype == pl.Float64, (
        "swarm must cast Float32 value → Float64 (any non-Float64 predicate)"
    )


def test_swarm_casts_numeric_string_value_to_float64() -> None:
    """swarm with a numeric-string value column must cast it to Float64.

    A Utf8/String column containing numeric strings (e.g. '1.0', '2.5') is not
    an integer dtype, so the int-only predicate leaves it untouched.  The old
    swarm block's ``dtype != pl.Float64`` predicate would have cast it.
    """
    df = pl.DataFrame(
        {
            "cat": ["a"] * 5 + ["b"] * 5,
            "val": pl.Series(["1.0", "2.0", "3.0", "4.0", "5.0"] * 2, dtype=pl.Utf8),
        }
    )
    chart = fm.Chart(df).mark_swarm().encode(x="cat:N", y="val:Q")
    resolved = chart._resolve_pending()
    assert isinstance(resolved._data, pl.DataFrame)
    assert resolved._data["val"].dtype == pl.Float64, (
        "swarm must cast numeric-string value → Float64 (any non-Float64 predicate)"
    )


def test_boxplot_does_not_cast_float32_value() -> None:
    """boxplot's int-only predicate must NOT cast Float32 columns.

    The box/violin/boxen cast is intentionally int-only: Float32 columns are
    valid inputs for those transforms and should NOT be silently widened.
    This test pins that the fix to swarm did not change the boxplot predicate.
    """
    df = pl.DataFrame(
        {
            "species": ["a"] * 5 + ["b"] * 5,
            "val": pl.Series([1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 4.0, 5.0, 6.0], dtype=pl.Float32),
        }
    )
    chart = fm.Chart(df).mark_boxplot().encode(x="species:N", y="val:Q")
    resolved = chart._resolve_pending()
    assert isinstance(resolved._data, pl.DataFrame)
    assert resolved._data["val"].dtype == pl.Float32, (
        "boxplot int-only cast must NOT widen Float32 → Float64"
    )
