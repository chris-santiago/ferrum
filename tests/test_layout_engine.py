"""Phase 6 layout engine — Python binding surface tests.

These validate the dict shape, error mapping, and end-to-end behavior
through the binding. Numeric arithmetic is exhaustively covered on the
Rust side (`cargo test -p ferrum-core layout`); these tests only confirm
the binding wires inputs and outputs correctly.
"""

import pytest

from ferrum._core import ChartSpec, compute_layout


def _spec():
    return ChartSpec(mark="point", x="a", y="b")


def test_compute_layout_returns_dict_with_expected_keys():
    r = compute_layout(
        _spec(),
        viewport=(600.0, 400.0),
        x_tick_labels=["0", "1", "2", "3"],
        y_tick_labels=["0", "5", "10"],
    )
    assert isinstance(r, dict)
    assert "viewport" in r
    assert "panels" in r
    assert "axes" in r
    # legend skipped when None
    assert "legend" not in r or r["legend"] is None
    # warnings skipped when empty
    assert "warnings" not in r or r["warnings"] == []


def test_single_chart_one_panel_two_axes():
    r = compute_layout(
        _spec(),
        viewport=(600.0, 400.0),
        x_tick_labels=["0", "1", "2"],
        y_tick_labels=["0", "5"],
    )
    assert len(r["panels"]) == 1
    assert len(r["axes"]) == 2
    p = r["panels"][0]
    assert p["plot_area"]["w"] > 0
    assert p["plot_area"]["h"] > 0
    assert p.get("facet_key") is None
    assert p["row"] == 0 and p["col"] == 0


def test_legend_present_when_entries_supplied():
    r = compute_layout(
        _spec(),
        viewport=(600.0, 400.0),
        x_tick_labels=["a", "b"],
        y_tick_labels=["0", "10"],
        legend_entries=[("setosa", "circle"), ("versicolor", "circle")],
    )
    assert r["legend"] is not None
    assert r["legend"]["orient"] == "right"
    assert len(r["legend"]["entries"]) == 2


def test_invalid_viewport_raises_value_error():
    with pytest.raises(ValueError, match="invalid viewport"):
        compute_layout(
            _spec(),
            viewport=(0.0, 400.0),
            x_tick_labels=["a"],
            y_tick_labels=["a"],
        )


def test_invalid_legend_orient_raises_value_error():
    with pytest.raises(ValueError, match="legend_orient"):
        compute_layout(
            _spec(),
            viewport=(600.0, 400.0),
            x_tick_labels=["a"],
            y_tick_labels=["a"],
            legend_orient="diagonal",
        )


def test_collision_triggers_rotation_in_axis_layout():
    # Many wide labels → x-axis collision → rotation.
    long_labels = [f"category_{i}" for i in range(20)]
    r = compute_layout(
        _spec(),
        viewport=(300.0, 400.0),  # narrow
        x_tick_labels=long_labels,
        y_tick_labels=["0", "10"],
    )
    x_axis = next(a for a in r["axes"] if a["orient"] == "bottom")
    angles = {tick["label_angle"] for tick in x_axis["ticks"]}
    assert -45.0 in angles or any(a != 0.0 for a in angles), (
        f"expected rotation; got angles {angles}"
    )
