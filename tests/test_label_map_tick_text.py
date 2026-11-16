"""``Axis(label_map=)`` relabels tick text only, applied by the renderer at draw time.

The data keeps its values: ``sort=``, ``BandScale(domain=)``, filters, legends and tooltips all
speak the data value, and a map on a ``+`` operand renames nothing of another layer. The map
reads a tick's formatted text (after ``label_format``), so it works on categorical, numeric and
temporal axes, per facet panel and composite leaf, and the interactive scene carries the same
tick text as the static SVG.

Tick text is read off the rendered SVG (and the interactive scene) as the text that follows an
axis tick line. The bars and their positions are compared with a twin chart whose data was
renamed by hand, so a relabel that also moved or dropped a bar cannot pass.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from ferrum.axis import Axis
from ferrum.configure import AxisConfig
from tests.test_null_group_warning import _NO_ROWS_TO_DRAW, _render_capturing

_CATS = ["a", "b", "c", "d"]
_DF = pl.DataFrame({"c": _CATS, "v": [2.0, 1.0, 4.0, 3.0]})
_RENAMED = _DF.with_columns(pl.col("c").replace({"b": "Bee"}))
_BEE = {"b": "Bee"}
_AXIS_GREY = "#6b7280"
_BAR_FILL = "#2563eb"

_NUMERIC = pl.DataFrame({"n": [1.0, 2.0, 3.0, 4.0], "v": [1.0, 2.0, 3.0, 4.0]})
_TEMPORAL = pl.DataFrame(
    {
        "d": pl.date_range(dt.date(2020, 1, 1), dt.date(2020, 1, 5), "1d", eager=True),
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)
_FACETED = pl.DataFrame(
    {
        "c": ["a", "b", "b", "c"],
        "v": [1.0, 2.0, 3.0, 4.0],
        "f": ["F1", "F1", "F2", "F2"],
    }
)

_TICK_PAIR = re.compile(r"<(line|text)\b([^>]*?)/?>(?:([^<]*)</text>)?")


def _attrs(raw: str) -> dict[str, str]:
    return dict(re.findall(r'([\w:-]+)="([^"]*)"', raw))


def _svg_ticks(svg: str, axis: str) -> list[str]:
    """Tick text of every ``axis`` axis in document order: the text right after a grey tick line."""
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.S)
    found = [(m.group(1), _attrs(m.group(2)), m.group(3) or "") for m in _TICK_PAIR.finditer(body)]
    out: list[str] = []
    for (tag, attrs, _), (next_tag, next_attrs, text) in zip(found, found[1:]):
        if tag != "line" or next_tag != "text" or attrs.get("stroke") != _AXIS_GREY:
            continue
        horizontal = attrs["y1"] == attrs["y2"]
        anchor = next_attrs.get("text-anchor")
        if (axis == "x" and not horizontal and anchor == "middle") or (
            axis == "y" and horizontal and anchor == "end"
        ):
            out.append(text)
    return out


def _scene_ticks(chart: Any, axis: str) -> list[str]:
    """The same tick text, read from the interactive scene's axis nodes."""
    out: list[str] = []
    for panel in json.loads(chart.interactive().scene_json)["panels"]:
        nodes = panel["axes"]
        for line, text in zip(nodes, nodes[1:]):
            if line["type"] != "line" or text["type"] != "text":
                continue
            colour = line["style"]["color"]
            if (colour["r"], colour["g"], colour["b"]) != (107, 114, 128):
                continue
            horizontal = line["y1"] == line["y2"]
            anchor = text["style"]["anchor"]
            if (axis == "x" and not horizontal and anchor == "middle") or (
                axis == "y" and horizontal and anchor == "end"
            ):
                out.append(text["content"])
    return out


def _svg_right_ticks(svg: str) -> list[str]:
    """Tick text of the secondary y axis: the start-anchored text after a horizontal grey tick."""
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.S)
    found = [(m.group(1), _attrs(m.group(2)), m.group(3) or "") for m in _TICK_PAIR.finditer(body)]
    return [
        text
        for (tag, attrs, _), (next_tag, next_attrs, text) in zip(found, found[1:])
        if tag == "line"
        and next_tag == "text"
        and attrs.get("stroke") == _AXIS_GREY
        and attrs["y1"] == attrs["y2"]
        and next_attrs.get("text-anchor") == "start"
    ]


def _scene_right_ticks(chart: Any) -> list[str]:
    """The secondary y axis's tick text from the interactive scene: every text on slot 1 or more."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text" and node.get("slot", 0) >= 1:
                out.append(node["content"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for panel in json.loads(chart.interactive().scene_json)["panels"]:
        walk(panel["axes"])
    return out


def _texts(svg: str) -> list[str]:
    return re.findall(r"<text\b[^>]*>([^<]*)</text>", svg)


def _bars(svg: str) -> list[tuple[str, ...]]:
    """Every bar's geometry, in draw order."""
    found = []
    for m in re.finditer(r"<rect\b([^>]*?)/?>", svg):
        attrs = _attrs(m.group(1))
        if attrs.get("fill") == _BAR_FILL:
            found.append(tuple(attrs[k] for k in ("x", "y", "width", "height")))
    return found


def _bar_chart(
    label_map: dict[str, str] | None = _BEE, frame: pl.DataFrame = _DF, **x_kwargs: Any
) -> fm.Chart:
    if label_map is not None:
        x_kwargs["axis"] = Axis(label_map=label_map)
    return fm.Chart(frame).mark_bar().encode(x=fm.X("c", **x_kwargs), y="v")


def _relabelled(ticks: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(t, t) for t in ticks]


# ---------------------------------------------------------------------------
# Categorical axes: the relabel is tick text only
# ---------------------------------------------------------------------------


def test_a_label_map_relabels_the_tick_and_leaves_the_drawing_as_the_renamed_data_draws_it():
    svg, shown = _render_capturing(_bar_chart().to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]
    assert svg == fm.Chart(_RENAMED).mark_bar().encode(x="c", y="v").to_svg()


def test_the_data_value_orders_the_bands_through_sort():
    chart = _bar_chart(sort=["d", "c", "b", "a"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["d", "c", "Bee", "a"]


def test_a_display_label_in_sort_is_unmatched_as_any_unknown_name_is():
    sort = ["d", "c", "Bee", "a"]
    mapped, mapped_shown = _render_capturing(_bar_chart(sort=sort).to_svg)
    plain, plain_shown = _render_capturing(_bar_chart(None, sort=sort).to_svg)
    assert mapped_shown == plain_shown
    assert _svg_ticks(mapped, "x") == _relabelled(_svg_ticks(plain, "x"), _BEE)
    assert _bars(mapped) == _bars(plain)


def test_a_band_domain_of_data_values_keeps_every_bar_and_warns_of_nothing():
    chart = _bar_chart(scale=fm.BandScale(domain=_CATS))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert len(_bars(svg)) == 4
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]


def test_a_filter_speaks_the_data_value():
    kept, shown = _render_capturing(
        _bar_chart().transform(fm.transform_filter("datum.c == 'b'")).to_svg
    )
    assert shown == []
    assert len(_bars(kept)) == 1
    assert _svg_ticks(kept, "x") == ["Bee"]

    emptied, shown = _render_capturing(
        _bar_chart().transform(fm.transform_filter("datum.c == 'Bee'")).to_svg
    )
    assert shown == [_NO_ROWS_TO_DRAW]
    assert _bars(emptied) == []


def test_a_colour_legend_on_the_mapped_column_reads_the_data_value():
    chart = (
        fm.Chart(_DF).mark_bar().encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v", color="c")
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]
    texts = _texts(svg)
    assert texts.count("Bee") == 1
    assert texts.count("b") == 1


def test_a_tooltip_carries_the_data_value():
    chart = (
        fm.Chart(_DF)
        .mark_bar()
        .encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v", tooltip=fm.Tooltip("c"))
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert re.findall(r"<title>([^<]*)</title>", svg) == ["c: a", "c: b", "c: c", "c: d"]


def test_a_swapping_map_swaps_tick_text_and_leaves_the_bars_in_data_order():
    swapped, shown = _render_capturing(_bar_chart({"a": "b", "b": "a"}).to_svg)
    assert shown == []
    assert _svg_ticks(swapped, "x") == ["b", "a", "c", "d"]
    assert _bars(swapped) == _bars(_bar_chart(None).to_svg())


@pytest.mark.parametrize("label_map", [{"zzz": "Z"}, {}], ids=["no-matching-key", "empty-map"])
def test_a_map_that_matches_no_tick_changes_nothing(label_map):
    svg, shown = _render_capturing(_bar_chart(label_map).to_svg)
    assert shown == []
    assert svg == _bar_chart(None).to_svg()


# ---------------------------------------------------------------------------
# Layering: the establishing layer's axis option reaches the panel axis
# ---------------------------------------------------------------------------


def _point_operand(label_map: dict[str, str] | None) -> fm.Chart:
    x_kwargs = {} if label_map is None else {"axis": Axis(label_map=label_map)}
    return fm.Chart(_DF).mark_point().encode(x=fm.X("c", **x_kwargs), y="v")


def test_an_operands_map_renames_nothing_of_the_host():
    chart = _bar_chart(None) + _point_operand(_BEE)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "b", "c", "d"]


def test_the_hosts_map_relabels_one_band_per_category():
    chart = _bar_chart() + _point_operand(None)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]


def test_an_operand_with_its_own_frame_does_not_add_a_band():
    other = (
        fm.Chart(pl.DataFrame({"c": ["b", "z"], "v": [1.0, 2.0]}))
        .mark_point()
        .encode(x=fm.X("c", axis=Axis(label_map={"z": "Zed"})), y="v")
    )
    svg, shown = _render_capturing((_bar_chart() + other).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d", "z"]


@pytest.mark.parametrize(
    ("host_map", "operand_map", "expected"),
    [
        (_BEE, None, ["a", "Bee", "c", "d"]),
        (None, _BEE, ["a", "b", "c", "d"]),
    ],
    ids=["host-has-the-map", "operand-has-the-map"],
)
def test_a_layer_chart_takes_the_establishing_members_map(host_map, operand_map, expected):
    chart = fm.LayerChart(_bar_chart(host_map), _point_operand(operand_map))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == expected


def test_chart_layer_relabels_the_shared_axis():
    chart = (
        fm.Chart(_DF)
        .encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
        .layer(fm.Layer(mark="bar"))
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]


# ---------------------------------------------------------------------------
# Numeric and temporal axes: the map reads the formatted tick text
# ---------------------------------------------------------------------------


def _numeric(label_map: dict[str, str] | None, **axis_kwargs: Any) -> fm.Chart:
    axis = Axis(label_map=label_map, **axis_kwargs)
    return fm.Chart(_NUMERIC).mark_point().encode(x=fm.X("n", axis=axis), y="v")


def test_a_numeric_axis_relabels_the_tick_whose_text_is_the_key():
    plain = _svg_ticks(_numeric(None).to_svg(), "x")
    assert "2" in plain
    svg, shown = _render_capturing(_numeric({"2": "two"}).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == _relabelled(plain, {"2": "two"})


def test_a_numeric_key_reads_the_text_after_label_format():
    plain = _svg_ticks(_numeric(None, label_format=".1f").to_svg(), "x")
    assert "2.0" in plain
    svg, shown = _render_capturing(_numeric({"2.0": "two"}, label_format=".1f").to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == _relabelled(plain, {"2.0": "two"})
    unformatted, _ = _render_capturing(_numeric({"2": "two"}, label_format=".1f").to_svg)
    assert _svg_ticks(unformatted, "x") == plain


def test_a_temporal_axis_relabels_a_tick_whose_formatted_text_is_the_key():
    def chart(label_map: dict[str, str] | None) -> fm.Chart:
        axis = Axis(label_map=label_map)
        return fm.Chart(_TEMPORAL).mark_point().encode(x=fm.X("d", axis=axis), y="v")

    plain = _svg_ticks(chart(None).to_svg(), "x")
    assert plain.count("12:00") >= 1
    svg, shown = _render_capturing(chart({"12:00": "noon"}).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == _relabelled(plain, {"12:00": "noon"})


# ---------------------------------------------------------------------------
# Construction: tick text is text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label_map", "got"),
    [({2: "two"}, "2: 'two'"), ({"b": None}, "'b': None"), ({"b": 3}, "'b': 3")],
    ids=["int-key", "none-value", "int-value"],
)
def test_a_non_string_key_or_value_is_refused_with_the_remedy(label_map, got):
    expected = (
        f"Axis(label_map=...): keys and values must be strings, got {got}; "
        "write the key as the tick text, e.g. {'2': 'two'}"
    )
    with pytest.raises(ValueError) as exc:
        Axis(label_map=label_map)
    assert str(exc.value) == expected


def test_the_chart_level_axis_config_refuses_the_same_maps():
    with pytest.raises(ValueError) as exc:
        AxisConfig(label_map={2: "two"})
    assert str(exc.value) == (
        "AxisConfig(label_map=...): keys and values must be strings, got 2: 'two'; "
        "write the key as the tick text, e.g. {'2': 'two'}"
    )


def test_the_remedy_the_refusal_names_renders():
    with pytest.raises(ValueError):
        Axis(label_map={2: "two"})
    plain = _svg_ticks(_numeric(None).to_svg(), "x")
    svg, shown = _render_capturing(_numeric({"2": "two"}).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == _relabelled(plain, {"2": "two"})


@pytest.mark.parametrize("owner", [Axis, AxisConfig], ids=["Axis", "AxisConfig"])
def test_a_map_that_is_not_a_dict_is_refused(owner):
    with pytest.raises(TypeError) as exc:
        owner(label_map=[("b", "Bee")])  # type: ignore[arg-type]
    assert str(exc.value) == (
        f"{owner.__name__}(label_map=...): expected a dict mapping tick text to display text, "
        "got list: [('b', 'Bee')]"
    )


@pytest.mark.parametrize("owner", [Axis, AxisConfig], ids=["Axis", "AxisConfig"])
def test_a_map_changed_after_construction_does_not_reach_the_wire(owner):
    given = {"b": "Bee"}
    held = owner(label_map=given)
    given["b"] = None  # type: ignore[assignment]
    given[2] = "two"  # type: ignore[index]
    assert held.label_map == {"b": "Bee"}
    assert held.to_dict()["label_map"] == {"b": "Bee"}
    if owner is AxisConfig:
        chart = fm.Chart(_DF).mark_bar().encode(x="c", y="v").configure(axis_x=held)
    else:
        chart = fm.Chart(_DF).mark_bar().encode(x=fm.X("c", axis=held), y="v")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]


# ---------------------------------------------------------------------------
# The wire: the map crosses as an axis key, and only when set
# ---------------------------------------------------------------------------


def test_to_dict_emits_a_set_map():
    assert Axis(label_map=_BEE).to_dict() == {"label_map": {"b": "Bee"}}


def test_to_dict_without_a_map_has_no_key_and_the_spec_json_is_unchanged():
    assert Axis(label_map=None).to_dict() == Axis().to_dict() == {}
    assert Axis(title="T").to_dict() == {"title": "T"}
    chart = fm.Chart(_DF).mark_bar().encode(x=fm.X("c", axis=Axis(title="T")), y="v")
    assert "label_map" not in chart.to_spec().to_json()


def test_a_set_map_crosses_in_the_spec_json():
    mapped = fm.Chart(_DF).mark_bar().encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
    assert '"label_map"' in mapped.to_spec().to_json()


# ---------------------------------------------------------------------------
# Chart-level configuration
# ---------------------------------------------------------------------------


def _plain_bars() -> fm.Chart:
    return _bar_chart(None)


def test_configure_axis_relabels_the_chart_axes():
    svg, shown = _render_capturing(_plain_bars().configure_axis(label_map=_BEE).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]


def test_configure_axis_x_config_relabels_the_x_ticks_only():
    chart = _plain_bars().configure(axis_x=AxisConfig(label_map={"a": "Ay", "2": "two"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["Ay", "b", "c", "d"]
    assert _svg_ticks(svg, "y") == _svg_ticks(_plain_bars().to_svg(), "y")


def test_a_per_channel_map_wins_over_a_chart_level_map():
    chart = _bar_chart(_BEE).configure_axis(label_map={"b": "Chart", "c": "See"})
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "c", "d"]


def test_configure_axis_y_config_relabels_the_y_ticks_only():
    chart = _plain_bars().configure(axis_y=AxisConfig(label_map={"2": "two", "b": "Bee"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "y") == ["4", "3.5", "3", "2.5", "two", "1.5", "1", "0.5", "0"]
    assert _svg_ticks(svg, "x") == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# A secondary y axis: an independent layer's own map, and the chart-level axis_y2
# ---------------------------------------------------------------------------

_TWO_Y = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "y2": [100.0, 200.0, 300.0]})
_RIGHT = {"200": "two hundred"}


def _two_y_chart(layer_map: dict[str, str] | None = None) -> fm.LayerChart:
    y2 = fm.Y("y2") if layer_map is None else fm.Y("y2", axis=Axis(label_map=layer_map))
    return fm.LayerChart(
        fm.Chart(_TWO_Y).mark_line().encode(x="x", y="y"),
        fm.Chart(_TWO_Y).mark_line().encode(x="x", y=y2),
        resolve={"y": "independent"},
    )


def _assert_right_axis_relabelled(chart: fm.LayerChart) -> None:
    plain = _two_y_chart()
    plain_right = _svg_right_ticks(plain.to_svg())
    assert "200" in plain_right
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    expected = _relabelled(plain_right, _RIGHT)
    assert _svg_right_ticks(svg) == expected
    assert _scene_right_ticks(chart) == expected
    assert _svg_ticks(svg, "y") == _svg_ticks(plain.to_svg(), "y")


def test_an_independent_y_layers_map_relabels_its_right_axis():
    _assert_right_axis_relabelled(_two_y_chart(_RIGHT))


def test_configure_axis_y2_relabels_the_right_axis():
    chart = _two_y_chart().configure(axis_y2=AxisConfig(label_map=_RIGHT))
    _assert_right_axis_relabelled(chart)


# ---------------------------------------------------------------------------
# Boundaries of the mapped column
# ---------------------------------------------------------------------------


def test_a_date_category_axis_takes_the_iso_label_as_the_key():
    frame = pl.DataFrame(
        {
            "d": pl.date_range(dt.date(2020, 1, 1), dt.date(2020, 1, 3), "1d", eager=True),
            "v": [1.0, 2.0, 3.0],
        }
    )

    def chart(**axis_kwargs: Any) -> fm.Chart:
        axis = Axis(**axis_kwargs)
        return fm.Chart(frame).mark_bar().encode(x=fm.X("d:O", axis=axis), y="v")

    plain, shown = _render_capturing(chart().to_svg)
    assert shown == []
    assert _svg_ticks(plain, "x") == ["2020-01-01", "2020-01-02", "2020-01-03"]
    svg, shown = _render_capturing(chart(label_map={"2020-01-02": "Jan 2"}).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["2020-01-01", "Jan 2", "2020-01-03"]


def test_a_zero_row_frame_with_a_map_draws_as_the_unmapped_one():
    empty = _DF.head(0)
    plain, plain_shown = _render_capturing(_bar_chart(None, frame=empty).to_svg)
    mapped, shown = _render_capturing(_bar_chart(frame=empty).to_svg)
    assert plain_shown == [_NO_ROWS_TO_DRAW]
    assert shown == [_NO_ROWS_TO_DRAW]
    assert mapped == plain


def test_a_null_in_the_mapped_column_is_dropped_and_warned_and_no_key_names_it():
    frame = pl.DataFrame({"c": ["a", "b", None], "v": [1.0, 2.0, 3.0]})
    label_map = {"b": "Bee", "None": "N", "null": "N", "": "E"}
    svg, shown = _render_capturing(_bar_chart(label_map, frame=frame).to_svg)
    assert _svg_ticks(svg, "x") == ["a", "Bee"]
    assert shown == [
        "mark_bar: 1 row has a null 'c' and belongs to no group, so it was dropped; "
        "fill or filter the nulls in 'c'"
    ]


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float32], ids=["Int64", "Float32"])
def test_an_integer_or_float32_numeric_axis_relabels_its_tick(dtype):
    frame = _NUMERIC.with_columns(pl.col("n").cast(dtype))

    def chart(label_map: dict[str, str]) -> fm.Chart:
        axis = Axis(label_map=label_map)
        return fm.Chart(frame).mark_point().encode(x=fm.X("n", axis=axis), y="v")

    plain = _svg_ticks(chart({}).to_svg(), "x")
    assert "2" in plain
    svg, shown = _render_capturing(chart({"2": "two"}).to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == _relabelled(plain, {"2": "two"})


# ---------------------------------------------------------------------------
# Every route: facets, composites, interactive
# ---------------------------------------------------------------------------


def test_independent_facet_axes_relabel_each_panel():
    chart = (
        fm.Chart(_FACETED)
        .mark_bar()
        .encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
        .facet("f")
        .share_scale(x="independent")
    )
    plain = chart.encode(x="c")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _svg_ticks(svg, "x") == ["a", "Bee", "Bee", "c"]
    assert _svg_ticks(plain.to_svg(), "x") == ["a", "b", "b", "c"]


@pytest.mark.parametrize("join", ["hconcat", "vconcat"])
def test_a_composite_relabels_each_leaf_axis(join):
    leaf = _bar_chart()
    chart = getattr(fm, join)(leaf, leaf)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    ticks = _svg_ticks(svg, "x")
    assert ticks
    assert "b" not in ticks
    assert ticks.count("Bee") == len(ticks) // 4


def _routes() -> dict[str, fm.Chart]:
    return {
        "categorical": _bar_chart(),
        "numeric": _numeric({"2": "two"}),
        "facet": (
            fm.Chart(_FACETED)
            .mark_bar()
            .encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
            .facet("f")
            .share_scale(x="independent")
        ),
        "hconcat": fm.hconcat(_bar_chart(), _bar_chart()),
        "plus": _bar_chart() + _point_operand(None),
        "chart_layer": (
            fm.Chart(_DF)
            .encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
            .layer(fm.Layer(mark="bar"))
        ),
        "layer_chart": fm.LayerChart(_bar_chart(), _point_operand(None)),
    }


_BEE_TICKS = ["a", "Bee", "c", "d"]
_EXPECTED_ROUTE_TICKS = {
    "categorical": _BEE_TICKS,
    "plus": _BEE_TICKS,
    "chart_layer": _BEE_TICKS,
    "layer_chart": _BEE_TICKS,
    "hconcat": _BEE_TICKS * 2,
    "facet": ["a", "Bee", "Bee", "c"],
}


@pytest.mark.parametrize("route", sorted(_routes()))
def test_the_interactive_scene_carries_the_static_tick_text(route):
    chart = _routes()[route]
    static = _svg_ticks(chart.to_svg(), "x")
    if route == "numeric":
        expected = _relabelled(_svg_ticks(_numeric(None).to_svg(), "x"), {"2": "two"})
    else:
        expected = _EXPECTED_ROUTE_TICKS[route]
    assert static == expected
    assert _scene_ticks(chart, "x") == expected


# ---------------------------------------------------------------------------
# The interactive scene tags each tick label with its physical axis
# ---------------------------------------------------------------------------


def _scene_text_nodes(node: Any) -> list[dict[str, Any]]:
    """Every text node anywhere in a scene (axes, marks, legend, titles), in document order."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "text":
            found.append(node)
        for value in node.values():
            found.extend(_scene_text_nodes(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_scene_text_nodes(value))
    return found


def _svg_tick_lines(svg: str, axis: str) -> list[str]:
    """Every text line of every ``axis`` tick: the text at a grey tick line's position, so a
    tick that wraps onto two lines (a date's month and day) contributes both."""
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.S)
    found = [(m.group(1), _attrs(m.group(2)), m.group(3) or "") for m in _TICK_PAIR.finditer(body)]
    at: set[float] = set()
    for tag, attrs, _ in found:
        if tag == "line" and attrs.get("stroke") == _AXIS_GREY:
            horizontal = attrs["y1"] == attrs["y2"]
            if (axis == "y") == horizontal:
                at.add(round(float(attrs["y1" if axis == "y" else "x1"]), 2))
    anchor = "end" if axis == "y" else "middle"
    out = []
    for tag, attrs, text in found:
        if tag != "text" or attrs.get("fill") != _AXIS_GREY or attrs.get("text-anchor") != anchor:
            continue
        if axis == "y":
            # A y label's baseline sits a third of its font size under its tick.
            on_tick = any(abs(float(attrs["y"]) - y) < 5 for y in at)
        else:
            on_tick = round(float(attrs["x"]), 2) in at
        if on_tick:
            out.append(text)
    return out


def _assert_ticks_carry_their_axis_tag(chart: Any) -> None:
    """The drawn tick text of each axis is exactly the nodes tagged with that axis.

    The tagged contents of each axis equal, in order, the static SVG's ticks of that axis (so
    the tagged count equals the static tick count), and no other text of the scene (title, legend entry,
    mark label) carries a ``tick_axis`` key.
    """
    svg = chart.to_svg()
    scene = json.loads(chart.interactive().scene_json)
    nodes = _scene_text_nodes(scene)
    for axis in ("x", "y"):
        tagged = [n["content"] for n in nodes if n.get("tick_axis") == axis]
        static = _svg_tick_lines(svg, axis)
        assert static, f"the fixture draws no {axis} ticks"
        assert tagged == static
        assert len(tagged) == len(static)
    assert {n["tick_axis"] for n in nodes if "tick_axis" in n} <= {"x", "y"}
    tagged_total = sum("tick_axis" in n for n in nodes)
    assert tagged_total == len(_svg_tick_lines(svg, "x")) + len(_svg_tick_lines(svg, "y"))


def _tag_fixtures() -> dict[str, Any]:
    def y_mapped() -> fm.Chart:
        axis = Axis(label_map={"2": "two"})
        return fm.Chart(_NUMERIC).mark_point().encode(x="n", y=fm.Y("v", axis=axis))

    def temporal() -> fm.Chart:
        axis = Axis(label_format="%b %d")
        return fm.Chart(_TEMPORAL).mark_point().encode(x=fm.X("d", axis=axis), y="v")

    def with_titles_legend_and_labels() -> fm.Chart:
        bars = fm.Chart(_DF).mark_bar().encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
        bars = bars.encode(color="c")
        labels = fm.Chart(_DF).mark_text().encode(x="c", y="v", text="c")
        return bars + labels

    return {
        "numeric-label-map": _numeric({"2": "two"}),
        "numeric-label-format": _numeric(None, label_format=".1f"),
        "temporal-label-format": temporal(),
        "categorical": _bar_chart(),
        "categorical-no-map": _bar_chart(None),
        "categorical-legend-and-labels": with_titles_legend_and_labels(),
        "y-label-map": y_mapped(),
        "facet": (
            fm.Chart(_FACETED)
            .mark_bar()
            .encode(x=fm.X("c", axis=Axis(label_map=_BEE)), y="v")
            .facet("f")
            .share_scale(x="independent")
        ),
        "hconcat": fm.hconcat(_bar_chart(), _bar_chart()),
    }


@pytest.mark.parametrize("name", sorted(_tag_fixtures()))
def test_every_drawn_tick_text_carries_its_axis_tag_in_the_interactive_scene(name):
    _assert_ticks_carry_their_axis_tag(_tag_fixtures()[name])


@pytest.mark.parametrize("name", sorted(_tag_fixtures()))
def test_the_tag_check_fails_when_the_tag_is_missing(name, monkeypatch):
    """The check discriminates: with ``tick_axis`` stripped from the scene it must fail."""
    real = _scene_text_nodes

    def stripped(node: Any) -> list[dict[str, Any]]:
        return [{k: v for k, v in n.items() if k != "tick_axis"} for n in real(node)]

    monkeypatch.setitem(globals(), "_scene_text_nodes", stripped)
    with pytest.raises(AssertionError):
        _assert_ticks_carry_their_axis_tag(_tag_fixtures()[name])
