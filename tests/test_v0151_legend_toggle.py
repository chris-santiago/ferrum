"""v0.15.1 audit: legend-toggle via add_params regression test.

Audit finding: a bind="legend" Selection attached via add_params() landed only
in _params, so scene.selections was empty and the WASM toggle_legend found
nothing — the legend click was a silent no-op.

Fix: add_params() now also registers Selection arguments into _selections so
the SelectionSpec is emitted into the rendered spec, in addition to the params
entry.  Dedup by name prevents double-registration when the same Selection is
later also passed to add_selection().
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum.selection import Selection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "cat": ["a", "b", "a"]})


def _selections_from_spec(chart: fm.Chart) -> list[dict]:
    """Return the parsed selections list from the chart spec JSON."""
    spec = chart.to_spec()
    spec_dict = json.loads(spec.to_json())
    raw = spec_dict.get("selections", [])
    # Already a list in the JSON; return as-is.
    return raw if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Core fix: Selection via add_params emits SelectionSpec
# ---------------------------------------------------------------------------


class TestLegendToggleViaAddParams:
    def test_selection_via_add_params_emits_selection_spec(self):
        """A Selection passed to add_params must appear in scene.selections."""
        df = _df()
        sel = fm.selection_point(bind="legend", name="legend_sel")
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x"), y=fm.Y("y"), color=fm.Color("cat"))
            .add_params(sel)
        )
        selections = _selections_from_spec(chart)
        assert len(selections) > 0, (
            "Expected at least one SelectionSpec in scene.selections when a "
            "Selection is passed via add_params(bind='legend'), got empty list"
        )
        names = [s.get("name") for s in selections]
        assert "legend_sel" in names, (
            f"Expected 'legend_sel' in scene.selections names, got {names!r}"
        )

    def test_selection_via_add_params_has_bind_in_params(self):
        """The bind='legend' attribute must survive into the params section."""
        df = _df()
        sel = fm.selection_point(bind="legend", name="legend_sel2")
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x"), y=fm.Y("y")).add_params(sel)
        spec = chart.to_spec()
        spec_dict = json.loads(spec.to_json())
        params_raw = spec_dict.get("params", [])
        params = params_raw if isinstance(params_raw, list) else []
        param_names = [p.get("name") for p in params]
        assert "legend_sel2" in param_names, (
            f"Expected 'legend_sel2' in params section, got {param_names!r}"
        )
        entry = next(p for p in params if p.get("name") == "legend_sel2")
        assert entry.get("bind") == "legend", (
            f"Expected bind='legend' in param entry, got {entry!r}"
        )

    def test_dedup_when_add_params_and_add_selection_both_used(self):
        """Same Selection via add_params then add_selection must not double-register."""
        df = _df()
        sel = fm.selection_point(bind="legend", name="dedup_sel")
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x"), y=fm.Y("y"))
            .add_params(sel)
            .add_selection(sel)
        )
        selections = _selections_from_spec(chart)
        names = [s.get("name") for s in selections]
        count = names.count("dedup_sel")
        assert count == 1, (
            f"Selection 'dedup_sel' registered {count} times in scene.selections "
            f"(expected exactly 1 after dedup)"
        )

    def test_add_params_with_plain_param_unchanged(self):
        """A plain fm.param() via add_params must not appear in _selections."""
        df = _df()
        k = fm.param("k_param", value=3)
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x"), y=fm.Y("y")).add_params(k)
        selections = _selections_from_spec(chart)
        names = [s.get("name") for s in selections]
        assert "k_param" not in names, (
            f"Plain fm.param() 'k_param' must not appear in scene.selections, but got {names!r}"
        )

    def test_add_params_still_rejects_non_parameter(self):
        """add_params() must still raise TypeError for non-Parameter args."""
        df = _df()
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x"), y=fm.Y("y"))
        with pytest.raises(TypeError, match="add_params\\(\\) expects Parameter"):
            chart.add_params("not_a_param")

    def test_selection_via_add_params_without_bind_emits_selection_spec(self):
        """A Selection without bind='legend' via add_params still emits SelectionSpec."""
        df = _df()
        sel = fm.selection_point(name="no_bind_sel")
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x"), y=fm.Y("y")).add_params(sel)
        selections = _selections_from_spec(chart)
        names = [s.get("name") for s in selections]
        assert "no_bind_sel" in names, f"Expected 'no_bind_sel' in scene.selections, got {names!r}"
