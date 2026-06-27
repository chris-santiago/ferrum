"""Drift-guard and byte-identity tests for the collapsed _scale_to_dict bridge.

Three test classes:

- ``TestScaleParity`` — drift guard: every ``*Scale`` class in ``ferrum._core``
  must have ``_to_scale_spec_dict`` and ``_scale_to_dict`` must return a dict
  with a ``"type"`` key for each one.
- ``TestByteIdentity`` — byte-identity guard: rebuild the 14 pre-change
  baseline charts and assert ``to_json()`` is unchanged.
- ``TestEncodeSmoke`` — regression: ``QuantileScale`` and ``ThresholdScale``
  passed to ``encode(scale=...)`` build and render without error (they raised
  ``TypeError`` before the bridge collapse).
"""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pytest

import ferrum as fr
import ferrum._core as fc
from ferrum.encoding._scale import _scale_to_dict

# ---------------------------------------------------------------------------
# Representative instances (one per *Scale class) used by the drift guard.
# When a new *Scale class is added without a _to_scale_spec_dict, the
# TestScaleParity.test_all_scale_classes_covered test will fail because the
# class name won't appear in _REPRESENTATIVE_INSTANCES.
# ---------------------------------------------------------------------------

_REPRESENTATIVE_INSTANCES: dict[str, object] = {
    "BandScale": fc.BandScale(),
    "BinOrdinalScale": fc.BinOrdinalScale(bins=[0.0, 0.5, 1.0]),
    "DivergingScale": fc.DivergingScale(),
    "LinearScale": fc.LinearScale(),
    "LogScale": fc.LogScale(),
    "OrdinalScale": fc.OrdinalScale(domain=["a", "b"]),
    "PointScale": fc.PointScale(),
    "PowScale": fc.PowScale(),
    "QuantileScale": fc.QuantileScale(domain=[1.0, 2.0, 3.0, 4.0], range=[0.0, 0.5, 1.0]),
    "QuantizeScale": fc.QuantizeScale(domain=[0.0, 1.0], range=["#f00", "#0f0", "#00f"]),
    "SequentialScale": fc.SequentialScale(),
    "SqrtScale": fc.SqrtScale(),
    "SymlogScale": fc.SymlogScale(),
    "ThresholdScale": fc.ThresholdScale(domain=[0.5], range=[0.0, 1.0]),
    "TimeScale": fc.TimeScale(domain=[0.0, 1000.0]),
}

_ALL_SCALE_NAMES = sorted(n for n in dir(fc) if n.endswith("Scale"))

# ---------------------------------------------------------------------------
# Baseline fixture path
# ---------------------------------------------------------------------------

_FIXTURE_PATH = pathlib.Path(__file__).parent / "_fixtures" / "scale_wire_baseline.json"


def _load_baseline() -> dict[str, str]:
    return json.loads(_FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Baseline chart builders: these MUST reproduce the exact chart constructions
# that generated the frozen tests/_fixtures/scale_wire_baseline.json (captured
# from pre-change `main`). The byte-identity guard compares this file's
# to_json() output against that fixture, so any divergence here fails loudly.
# ---------------------------------------------------------------------------


def _build_baseline_charts() -> dict[str, "fr.Chart"]:
    num = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "c": [0.1, 0.4, 0.7, 0.95],
        }
    )
    cat = pl.DataFrame({"k": ["x", "y", "z", "w"], "v": [1.0, 2.0, 3.0, 4.0]})
    tmp = pl.DataFrame(
        {
            "t": ["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("t").str.to_datetime())

    return {
        "linear": fr.Chart(num)
        .mark_point()
        .encode(x=fr.X("a", scale=fr.LinearScale(domain=[0.0, 5.0])), y="b"),
        "log": fr.Chart(num)
        .mark_point()
        .encode(x="a", y=fr.Y("b", scale=fr.LogScale(domain=[1.0, 100.0]))),
        "pow": fr.Chart(num)
        .mark_point()
        .encode(x=fr.X("a", scale=fr.PowScale(domain=[0.0, 5.0], exponent=2.0)), y="b"),
        "sqrt": fr.Chart(num)
        .mark_point()
        .encode(x=fr.X("a", scale=fr.SqrtScale(domain=[0.0, 5.0])), y="b"),
        "symlog": fr.Chart(num)
        .mark_point()
        .encode(x=fr.X("a", scale=fr.SymlogScale(domain=[0.0, 5.0], constant=1.0)), y="b"),
        "time": fr.Chart(tmp)
        .mark_point()
        .encode(
            x=fr.X("t", scale=fr.TimeScale(domain=[1.5778368e12, 1.6243200e12])),
            y="v",
        ),
        "utc": fr.Chart(tmp)
        .mark_point()
        .encode(
            x=fr.X("t", scale=fr.TimeScale(domain=[1.5778368e12, 1.6243200e12], utc=True)),
            y="v",
        ),
        "band": fr.Chart(cat)
        .mark_bar()
        .encode(
            x=fr.X("k", scale=fr.BandScale(domain=["x", "y", "z", "w"], padding=0.1)),
            y="v",
        ),
        "point": fr.Chart(cat)
        .mark_point()
        .encode(
            x=fr.X("k", scale=fr.PointScale(domain=["x", "y", "z", "w"], padding=0.5)),
            y="v",
        ),
        "ordinal": fr.Chart(cat)
        .mark_point()
        .encode(
            x=fr.X(
                "k",
                scale=fr.OrdinalScale(domain=["x", "y", "z", "w"], range=[0.0, 1.0, 2.0, 3.0]),
            ),
            y="v",
        ),
        "sequential": fr.Chart(num)
        .mark_point()
        .encode(
            x="a",
            y="b",
            color=fr.Color("c", scale=fr.SequentialScale(scheme="viridis")),
        ),
        "diverging": fr.Chart(num)
        .mark_point()
        .encode(
            x="a",
            y="b",
            color=fr.Color("c", scale=fr.DivergingScale(scheme="redblue", domain_mid=0.5)),
        ),
        "quantize": fr.Chart(num)
        .mark_point()
        .encode(
            x="a",
            y="b",
            color=fr.Color(
                "c",
                scale=fr.QuantizeScale(domain=[0.0, 1.0], range=["#f00", "#0f0", "#00f"]),
            ),
        ),
        "bin_ordinal": fr.Chart(num)
        .mark_point()
        .encode(
            x="a",
            y="b",
            color=fr.Color(
                "c",
                scale=fr.BinOrdinalScale(bins=[0.0, 0.5, 1.0], scheme="viridis"),
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScaleParity:
    """Drift guard: every *Scale class in ferrum._core must map through _scale_to_dict."""

    def test_all_scale_classes_covered(self):
        """_REPRESENTATIVE_INSTANCES covers every *Scale class in ferrum._core.

        Fails when a new *Scale class is added without adding it here and giving
        it a _to_scale_spec_dict method.
        """
        missing = [n for n in _ALL_SCALE_NAMES if n not in _REPRESENTATIVE_INSTANCES]
        assert not missing, (
            f"No representative instance for scale class(es): {missing}. "
            "Add entries to _REPRESENTATIVE_INSTANCES in test_scale_spec_parity.py "
            "and implement _to_scale_spec_dict on the Rust pyclass."
        )

    @pytest.mark.parametrize("name", _ALL_SCALE_NAMES)
    def test_scale_class_has_delegation_method(self, name: str):
        """Every *Scale pyclass must expose _to_scale_spec_dict (the delegation hook)."""
        cls = getattr(fc, name)
        assert hasattr(cls, "_to_scale_spec_dict"), (
            f"{name} lacks _to_scale_spec_dict — implement it on the Rust pyclass"
        )

    @pytest.mark.parametrize("name", _ALL_SCALE_NAMES)
    def test_scale_to_dict_returns_typed_dict(self, name: str):
        """_scale_to_dict(instance) returns a dict with a 'type' key for every class."""
        if name not in _REPRESENTATIVE_INSTANCES:
            pytest.skip(f"No representative instance for {name}")
        s = _REPRESENTATIVE_INSTANCES[name]
        d = _scale_to_dict(s)
        assert isinstance(d, dict), (
            f"_scale_to_dict({name}) returned {type(d).__name__!r}, expected dict"
        )
        assert "type" in d, f"_scale_to_dict({name}) dict missing 'type' key: {d!r}"


class TestByteIdentity:
    """to_json() is byte-identical to the pre-change baseline for all 14 scale types."""

    @pytest.mark.parametrize(
        "name",
        [
            "linear",
            "log",
            "pow",
            "sqrt",
            "symlog",
            "time",
            "utc",
            "band",
            "point",
            "ordinal",
            "sequential",
            "diverging",
            "quantize",
            "bin_ordinal",
        ],
    )
    def test_to_json_matches_baseline(self, name: str):
        """Chart.to_json() is byte-identical to the frozen pre-change baseline."""
        baseline = _load_baseline()
        charts = _build_baseline_charts()
        actual = charts[name].to_json()
        expected = baseline[name]
        assert actual == expected, (
            f"to_json() for '{name}' scale differs from baseline.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


class TestEncodeSmoke:
    """QuantileScale and ThresholdScale work in encode(scale=...) end-to-end.

    Pre-collapse these raised TypeError because _scale_to_dict returned the
    raw pyclass object (not serialisable). After the collapse, _to_scale_spec_dict
    is called and they produce valid wire dicts.
    """

    @pytest.fixture()
    def num_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [10.0, 20.0, 30.0, 40.0],
                "c": [0.1, 0.4, 0.7, 0.95],
            }
        )

    def test_quantile_scale_builds_and_renders(self, num_df: pl.DataFrame):
        chart = (
            fr.Chart(num_df)
            .mark_point()
            .encode(
                x="a",
                y="b",
                color=fr.Color(
                    "c",
                    scale=fr.QuantileScale(domain=[0.1, 0.4, 0.7, 0.95], range=[0.0, 0.5, 1.0]),
                ),
            )
        )
        json_str = chart.to_json()
        assert '"quantile"' in json_str
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg

    def test_threshold_scale_builds_and_renders(self, num_df: pl.DataFrame):
        chart = (
            fr.Chart(num_df)
            .mark_point()
            .encode(
                x="a",
                y="b",
                color=fr.Color(
                    "c",
                    scale=fr.ThresholdScale(domain=[0.3, 0.6], range=[0.0, 0.5, 1.0]),
                ),
            )
        )
        json_str = chart.to_json()
        assert '"threshold"' in json_str
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg

    def test_quantile_scale_in_x_channel(self, num_df: pl.DataFrame):
        """QuantileScale in a positional channel builds without error."""
        chart = (
            fr.Chart(num_df)
            .mark_point()
            .encode(
                x=fr.X(
                    "a",
                    scale=fr.QuantileScale(domain=[1.0, 2.0, 3.0, 4.0], range=[0.0, 0.5, 1.0]),
                ),
                y="b",
            )
        )
        assert chart.to_json() is not None
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg

    def test_threshold_scale_in_x_channel(self, num_df: pl.DataFrame):
        """ThresholdScale in a positional channel builds without error."""
        chart = (
            fr.Chart(num_df)
            .mark_point()
            .encode(
                x=fr.X(
                    "a",
                    scale=fr.ThresholdScale(domain=[2.0, 3.0], range=[0.0, 0.5, 1.0]),
                ),
                y="b",
            )
        )
        assert chart.to_json() is not None
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg
