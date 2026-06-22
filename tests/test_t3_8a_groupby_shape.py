"""Regression tests for T3.8a: groupby on Bin/Kde/Kde2D/Smooth accepts Vec<String>.

The Rust migration changed the groupby field from Option<String> to Vec<String>.
The PyO3 constructor now accepts None, a str (legacy alias for ["x"]), or a list[str].
These tests assert:
  1. Multi-column list is accepted (the load-bearing new case).
  2. Single string (legacy) still works.
  3. Single string and equivalent single-element list produce identical JSON (byte-stable
     only after round-trip, since the Rust side normalises str -> [str]).
"""

import json

from ferrum._core import Bin, ChartSpec, Kde, Kde2D, Smooth


# ---------------------------------------------------------------------------
# Bin
# ---------------------------------------------------------------------------


def test_bin_groupby_list_multicolumn():
    spec = Bin(field="price", bin_count=10, groupby=["region", "category"])
    cs = ChartSpec(mark="bar", x="price", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_bin_groupby_single_string_legacy():
    spec = Bin(field="price", bin_count=10, groupby="region")
    cs = ChartSpec(mark="bar", x="price", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_bin_groupby_string_equals_single_element_list():
    """str alias and [str] must produce identical serialised output."""
    cs_str = ChartSpec(mark="bar", x="price", transforms=[Bin(field="price", groupby="region")])
    cs_list = ChartSpec(mark="bar", x="price", transforms=[Bin(field="price", groupby=["region"])])
    assert json.loads(cs_str.to_json()) == json.loads(cs_list.to_json())


def test_bin_groupby_none():
    spec = Bin(field="price", bin_count=10, groupby=None)
    cs = ChartSpec(mark="bar", x="price", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


# ---------------------------------------------------------------------------
# Kde
# ---------------------------------------------------------------------------


def test_kde_groupby_list_multicolumn():
    spec = Kde(field="value", groupby=["group", "sub"])
    cs = ChartSpec(mark="line", x="value", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_kde_groupby_single_string_legacy():
    spec = Kde(field="value", groupby="group")
    cs = ChartSpec(mark="line", x="value", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_kde_groupby_string_equals_single_element_list():
    cs_str = ChartSpec(mark="line", x="value", transforms=[Kde(field="value", groupby="group")])
    cs_list = ChartSpec(mark="line", x="value", transforms=[Kde(field="value", groupby=["group"])])
    assert json.loads(cs_str.to_json()) == json.loads(cs_list.to_json())


# ---------------------------------------------------------------------------
# Kde2D
# ---------------------------------------------------------------------------


def test_kde2d_groupby_list_multicolumn():
    spec = Kde2D(x="x", y="y", groupby=["group", "sub"])
    cs = ChartSpec(mark="rect", x="x", y="y", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_kde2d_groupby_single_string_legacy():
    spec = Kde2D(x="x", y="y", groupby="group")
    cs = ChartSpec(mark="rect", x="x", y="y", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_kde2d_groupby_string_equals_single_element_list():
    cs_str = ChartSpec(mark="rect", x="x", y="y", transforms=[Kde2D(x="x", y="y", groupby="group")])
    cs_list = ChartSpec(
        mark="rect", x="x", y="y", transforms=[Kde2D(x="x", y="y", groupby=["group"])]
    )
    assert json.loads(cs_str.to_json()) == json.loads(cs_list.to_json())


# ---------------------------------------------------------------------------
# Smooth
# ---------------------------------------------------------------------------


def test_smooth_groupby_list_multicolumn():
    spec = Smooth(x="x", y="y", method="lm", groupby=["group", "sub"])
    cs = ChartSpec(mark="line", x="x", y="y", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_smooth_groupby_single_string_legacy():
    spec = Smooth(x="x", y="y", method="lm", groupby="group")
    cs = ChartSpec(mark="line", x="x", y="y", transforms=[spec])
    rt = ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_smooth_groupby_string_equals_single_element_list():
    cs_str = ChartSpec(
        mark="line", x="x", y="y", transforms=[Smooth(x="x", y="y", groupby="group")]
    )
    cs_list = ChartSpec(
        mark="line", x="x", y="y", transforms=[Smooth(x="x", y="y", groupby=["group"])]
    )
    assert json.loads(cs_str.to_json()) == json.loads(cs_list.to_json())
