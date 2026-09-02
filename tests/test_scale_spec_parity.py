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

Regenerating the baseline
--------------------------
``tests/_fixtures/scale_wire_baseline.json`` is regenerated ONLY via
``python scripts/regen-scale-wire-baseline.py <ref>`` (default ref: the
latest ``v*`` tag) — never by hand-editing the file or copying today's
``to_json()`` output out of a passing test run. The script builds an
isolated ``git worktree`` checkout of *ref*, compiles the Rust extension
there, and captures ``_build_baseline_charts()`` from THAT ref's copy of
this module. Regenerating from the current working tree instead would make
``TestByteIdentity`` a tautology: the guard exists to catch the wire format
silently drifting out from under a refactor, and a working-tree regen
would just rebaseline against whatever the (possibly already-drifted)
working tree currently emits, defeating the guard's purpose. See the
script's module docstring for the full rationale and build approach.
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
    """Load the fixture, dropping its ``_provenance`` metadata key.

    ``_provenance`` (``{"ref", "captured", "script"}``, written by
    ``scripts/regen-scale-wire-baseline.py``) is bookkeeping, not one of the
    14 scale-name payload entries ``TestByteIdentity`` compares against --
    strip it here so any future dict-shaped consumer of this loader never
    trips over it.
    """
    payload = json.loads(_FIXTURE_PATH.read_text())
    payload.pop("_provenance", None)
    return payload


# ---------------------------------------------------------------------------
# Baseline chart builders: these MUST reproduce the exact chart constructions
# that generated the frozen tests/_fixtures/scale_wire_baseline.json (captured
# from the ref recorded in the fixture's "_provenance" key via
# scripts/regen-scale-wire-baseline.py — see that script and this module's
# docstring for why regeneration never runs against the working tree). The
# byte-identity guard compares this file's to_json() output against that
# fixture, so any divergence here fails loudly. If you edit a chart
# construction below, the fixture must be regenerated via the script (not
# hand-edited) so "baseline" keeps meaning "what a specific frozen ref
# produced."
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


class TestAppearanceScaleWireGate:
    """StrokeOpacity/StrokeDash ``scale=`` (and ``title=``) now reach the wire.

    Batch-A appearance-resolution spec §4.3 (2026-08-28): these two channels
    moved from ``APPEARANCE_BASE`` to ``APPEARANCE_FULL`` in ``_honored.py``
    so ``scale=``/``title=`` serialize instead of warn-and-drop. Task 6
    (concurrent, Rust) is the consumer — these tests pin the Python-side wire
    contract independent of whether the Rust scale resolver has landed yet;
    a render must not error either way.
    """

    @pytest.fixture()
    def opacity_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "sw": [0.1, 0.4, 0.7, 0.95],
                "cat": ["a", "b", "a", "b"],
            }
        )

    def test_stroke_opacity_scale_serializes(self, opacity_df: pl.DataFrame):
        """StrokeOpacity(scale=LinearScale(...)) reaches the wire, not dropped."""
        chart = (
            fr.Chart(opacity_df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                stroke_opacity=fr.StrokeOpacity(
                    "sw",
                    scale=fr.LinearScale(domain=[0.0, 1.0], range=[0.2, 1.0]),
                    title="Opacity",
                ),
            )
        )
        json_str = chart.to_json()
        assert '"stroke_opacity"' in json_str
        assert '"scale":{"type":"linear"' in json_str
        assert '"title":"Opacity"' in json_str
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg

    def test_stroke_dash_scale_serializes(self, opacity_df: pl.DataFrame):
        """StrokeDash(scale=OrdinalScale(...)) reaches the wire, not dropped."""
        chart = (
            fr.Chart(opacity_df)
            .mark_line()
            .encode(
                x="x",
                y="y",
                color="cat",
                stroke_dash=fr.StrokeDash(
                    "cat",
                    scale=fr.OrdinalScale(domain=["a", "b"]),
                    title="Category",
                ),
            )
        )
        json_str = chart.to_json()
        assert '"stroke_dash"' in json_str
        assert '"scale":{"type":"ordinal"' in json_str
        assert '"title":"Category"' in json_str
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg

    def test_stroke_opacity_no_scale_still_omits_scale_key(self, opacity_df: pl.DataFrame):
        """Absent-case byte-identity: no scale= means no "scale" key on the wire."""
        chart = (
            fr.Chart(opacity_df)
            .mark_point()
            .encode(x="x", y="y", stroke_opacity=fr.StrokeOpacity("sw"))
        )
        json_str = chart.to_json()
        assert '"stroke_opacity":{"field":"sw"}' in json_str

    def test_stroke_dash_no_scale_still_omits_scale_key(self, opacity_df: pl.DataFrame):
        """Absent-case byte-identity: no scale= means no "scale" key on the wire."""
        chart = (
            fr.Chart(opacity_df).mark_line().encode(x="x", y="y", stroke_dash=fr.StrokeDash("cat"))
        )
        json_str = chart.to_json()
        assert '"stroke_dash":{"field":"cat"}' in json_str

    def test_override_flat_path_widening_pins_stroke_opacity_and_stroke_dash(
        self, opacity_df: pl.DataFrame
    ):
        """Emergent widening of Chart.override()'s flat-path registry, pinned.

        ``_override_apply._scale_bearing_channels()`` (src/ferrum/_override_apply.py)
        derives its channel list from each channel class's ``_honored_kwargs`` at
        import time. Moving StrokeOpacity/StrokeDash onto ``APPEARANCE_FULL``
        automatically widened that registry: ``stroke_opacity_scale_*`` and
        ``stroke_dash_scale_*`` flat-override paths (e.g.
        ``.override(stroke_opacity_scale_range=[...])``) became valid, the same
        way ``fill_opacity_scale_*`` already was. This is coherent (the registry
        is meant to track ``_honored_kwargs`` exactly) but was untested before
        this pin — assert both new paths resolve, build a correctly shaped
        payload, and round-trip through a real ``.override()`` render without
        error.
        """
        from ferrum import _override_apply as oa

        opacity_resolved = oa.resolve("stroke_opacity_scale_domain")
        assert opacity_resolved is not None
        assert opacity_resolved.target is oa.Target.ENCODING_SCALE
        assert opacity_resolved.location.target_key == "stroke_opacity"
        assert opacity_resolved.location.leaf == "domain"

        dash_resolved = oa.resolve("stroke_dash_scale_range")
        assert dash_resolved is not None
        assert dash_resolved.target is oa.Target.ENCODING_SCALE
        assert dash_resolved.location.target_key == "stroke_dash"
        assert dash_resolved.location.leaf == "range"

        payload = oa.build_payload({"stroke_opacity_scale_domain": [0.0, 1.0]})
        assert payload.encoding == {"stroke_opacity": {"scale": {"domain": [0.0, 1.0]}}}

        chart = (
            fr.Chart(opacity_df)
            .mark_point()
            .encode(x="x", y="y", stroke_opacity=fr.StrokeOpacity("sw"))
            .override(stroke_opacity_scale_domain=[0.0, 1.0], stroke_opacity_scale_range=[0.2, 1.0])
        )
        svg = chart.to_svg()
        assert isinstance(svg, str) and len(svg) > 0 and "<svg" in svg


class TestPositionalExtent:
    """Regression guard for the SPEC-04 positional-channel truncation fix (issue #38)
    and the same bug class in DivergingScale (issue #40).

    Before the fix in positional.rs, QuantileScale and ThresholdScale on x/y channels
    were routed through the domain-as-extent arm, which collapsed the axis to domain[0..1]
    and silently dropped data points outside that unit interval.  Only 2 of 4 marks
    rendered.  These tests lock the corrected behavior at exactly 4/4.

    DivergingScale(domain=[lo, mid, hi]) on a positional channel hits the same
    domain-as-extent arm: the axis truncates to [lo, mid] instead of [lo, hi],
    dropping marks above mid (issue #40).
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

    def test_diverging_positional_all_marks_render(self, num_df: pl.DataFrame):
        """Regression: DivergingScale(domain=[lo, mid, hi]) positionally routed
        through the domain-as-extent arm, truncating the axis to [lo, mid] and
        silently dropping marks above mid (issue #40). Before this fix the
        count was 2/4; it must be 4/4 now.
        """
        chart = (
            fr.Chart(num_df)
            .mark_point()
            .encode(
                x=fr.X(
                    "a",
                    scale=fr.DivergingScale(domain=[1.0, 2.5, 4.0]),
                ),
                y="b",
            )
        )
        svg = chart.to_svg()
        mark_count = len(re.findall(r"<circle|<path", svg))
        assert mark_count == 4, (
            f"Expected 4 rendered marks (one per row), got {mark_count}. "
            "DivergingScale 3-element domain on x truncated to [lo, mid] — "
            "issue #40 regression."
        )
