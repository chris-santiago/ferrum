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
import re

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
# Channel-routing constants for the minimal-chart drift helper
# ---------------------------------------------------------------------------

# These scale types only make sense on a color channel.
_COLOR_ONLY_SCALES: frozenset[str] = frozenset(
    {"BinOrdinalScale", "DivergingScale", "QuantizeScale", "SequentialScale"}
)
# These scale types require categorical (string) x data.
_CATEGORICAL_X_SCALES: frozenset[str] = frozenset({"BandScale", "OrdinalScale", "PointScale"})
# These scale types require datetime x data.
_TIME_X_SCALES: frozenset[str] = frozenset({"TimeScale"})


def _build_minimal_chart(name: str, instance: object) -> "fr.Chart":
    """Build the smallest valid chart that uses *instance* in the right encoding channel.

    Sequential/Diverging/Quantize/BinOrdinal are color-only; Band/Ordinal/Point need
    categorical x data; Time needs datetime x data; all others go on numeric x.
    """
    _num = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [0.1, 0.9]})
    _cat = pl.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]})
    _tmp = pl.DataFrame({"t": ["2020-01-01", "2021-01-01"], "v": [1.0, 2.0]}).with_columns(
        pl.col("t").str.to_datetime()
    )

    if name in _COLOR_ONLY_SCALES:
        return fr.Chart(_num).mark_point().encode(x="a", y="b", color=fr.Color("c", scale=instance))
    if name in _CATEGORICAL_X_SCALES:
        return fr.Chart(_cat).mark_point().encode(x=fr.X("k", scale=instance), y="v")
    if name in _TIME_X_SCALES:
        return fr.Chart(_tmp).mark_point().encode(x=fr.X("t", scale=instance), y="v")
    return fr.Chart(_num).mark_point().encode(x=fr.X("a", scale=instance), y="b")


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
        """_scale_to_dict(instance) yields a typed dict that round-trips into the wire format.

        For each representative instance this also builds a minimal one-channel chart and
        asserts chart.to_json() succeeds with the scale type present in the output — a DEEP
        check that auto-extends to any future 16th scale class added to
        _REPRESENTATIVE_INSTANCES without needing a second edit here.
        """
        # test_all_scale_classes_covered already ensures every name has an entry; no skip needed.
        s = _REPRESENTATIVE_INSTANCES[name]
        d = _scale_to_dict(s)
        assert isinstance(d, dict), (
            f"_scale_to_dict({name}) returned {type(d).__name__!r}, expected dict"
        )
        assert "type" in d, f"_scale_to_dict({name}) dict missing 'type' key: {d!r}"

        chart = _build_minimal_chart(name, s)
        json_str = chart.to_json()
        assert json_str is not None, f"chart.to_json() returned None for {name}"
        assert f'"{d["type"]}"' in json_str, (
            f"Scale type {d['type']!r} not found in chart.to_json() for {name}.\n"
            f"Wire dict: {d!r}\n"
            f"JSON: {json_str!r}"
        )


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
        """QuantileScale in a positional channel renders all data marks."""
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
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg
        mark_count = len(re.findall(r"<circle|<path", svg))
        assert mark_count == 4, f"Expected 4 rendered marks, got {mark_count}"

    def test_threshold_scale_in_x_channel(self, num_df: pl.DataFrame):
        """ThresholdScale in a positional channel renders all data marks."""
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
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg
        mark_count = len(re.findall(r"<circle|<path", svg))
        assert mark_count == 4, f"Expected 4 rendered marks, got {mark_count}"


class TestPositionalExtent:
    """Regression guard for the SPEC-04 positional-channel truncation fix (issue #38).

    Before the fix in positional.rs, QuantileScale and ThresholdScale on x/y channels
    were routed through the domain-as-extent arm, which collapsed the axis to domain[0..1]
    and silently dropped data points outside that unit interval.  Only 2 of 4 marks
    rendered.  These tests lock the corrected behavior at exactly 4/4.
    """

    @pytest.fixture()
    def num_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def test_quantile_positional_all_marks_render(self, num_df: pl.DataFrame):
        """Regression: SPEC-04 positionally routed QuantileScale through the
        domain-as-extent arm, collapsing the axis to domain[0..1] and dropping
        data points (caught by design review, issue #38). Before this fix the
        count was 2/4; it must be 4/4 now.
        """
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
        svg = chart.to_svg()
        mark_count = len(re.findall(r"<circle|<path", svg))
        assert mark_count == 4, (
            f"Expected 4 rendered marks (one per row), got {mark_count}. "
            "SPEC-04 positional truncation regression: QuantileScale on x may be "
            "routing through domain-as-extent again (issue #38)."
        )

    def test_threshold_positional_all_marks_render(self, num_df: pl.DataFrame):
        """Regression: SPEC-04 positionally routed ThresholdScale through the
        domain-as-extent arm, collapsing the axis to domain[0..1] and dropping
        data points (caught by design review, issue #38). Before this fix the
        count was 2/4; it must be 4/4 now.
        """
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
        svg = chart.to_svg()
        mark_count = len(re.findall(r"<circle|<path", svg))
        assert mark_count == 4, (
            f"Expected 4 rendered marks (one per row), got {mark_count}. "
            "SPEC-04 positional truncation regression: ThresholdScale on x may be "
            "routing through domain-as-extent again (issue #38)."
        )
