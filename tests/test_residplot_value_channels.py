"""residplot's value channels reach its residual points.

A value channel binding a data column that is neither the predictor nor the
response -- ``size=``, ``opacity=``, ``fill_opacity=``, ``stroke_opacity=``,
``stroke_width=``, ``tooltip=``, ``key=``, ``href=``, ``description=`` --
reaches the residual points on every door: a keyword, ``encode=``, and a
later ``.encode()``. The residual transform (``Smooth``/``Robust`` with
``output="residuals"``) carries every other column of the input row it
came from, beside ``x``, ``residual`` and the optional zero-reference/
metrics columns, the way a row-preserving statistic does.

On residplot's layered form (any overlay requested: the default), a
keyword or ``encode=`` value channel binds the point layer alone -- the
zero-reference rule and the corner metrics draw one mark for the whole
fit, and the lowess overlay reads its own, separate fit-grid batch, so
none of them ever inherit a per-row binding meant for the points. A LATER
``.encode()`` binds the whole chart, so every layer inherits it: the
points draw it, and the summary layers and the lowess overlay, which lack
it, drop it and warn, the way a layered statistic's missing column
always does.

The caller's own data is never renamed: every door that reads the
caller's rows -- a later binding, a
``.transform()`` step, an attached layer, a facet -- reads the caller's
own column under its own name. When the data already uses one of the
transform's own default names (``residual``, ``_ref_zero``, a metrics
name, or ``x`` when it is not the predictor), the COMPUTED column takes a
fresh name instead, invisibly.

Flat residplot (no overlay: its one mark reads x/y straight from the
chart-level encoding) refuses a post-build x/y binding naming another
column of the data, naming the call as the remedy, since drawing that
column raw -- or the fit's residuals at its positions -- would never be
what a refit against it looks like.

Grouping keywords (``color=``, ``fill=``, ``detail=``, ``shape=``,
``stroke_dash=``, ``label=``) are unaffected: a grouping field is read by
name directly, as a group key, never carried as a pass-through extra.
"""

from __future__ import annotations

import inspect
import json
import warnings
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum._scene import _render_scene
from ferrum._warn import reset_warnings

# ---------------------------------------------------------------------------
# Fixtures and scene-reading helpers
# ---------------------------------------------------------------------------


def _frame(**extra: Any) -> pl.DataFrame:
    """Six rows: a predictor ``v``, a response ``w``, a numeric ``s`` and a
    string ``id`` -- the two columns every value-channel case below binds."""
    columns = {
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "w": [2.0, 3.0, 1.0, 4.0, 3.5, 5.0],
        "s": [1.0, 5.0, 10.0, 2.0, 3.0, 4.0],
        "id": ["a", "b", "c", "d", "e", "f"],
    }
    columns.update(extra)
    return pl.DataFrame(columns)


#: residplot's flat form: no overlay requested (its default is layered --
#: a point layer, a dashed zero-reference rule, and a metrics-text layer,
#: all reading the SAME residual batch), so its one mark reads x/y
#: straight from the chart-level encoding instead.
def _flat_kwargs() -> dict:
    return {"zero_line": False, "show_metrics": False}


def _scene_marks(chart: Any) -> list[dict]:
    """Every mark batch of *chart*'s one (unfaceted) panel, in layer order."""
    scene_json, _packed = _render_scene(chart)
    return json.loads(scene_json)["panels"][0]["marks"]


def _point_batch(chart: Any) -> dict:
    """The point layer's own mark batch -- always first, on both forms."""
    return _scene_marks(chart)[0]


def _node_cy(node: dict) -> float:
    """A point node's vertical center, whichever key its own symbol uses.

    ``shape=`` draws a different symbol per group (a circle, a square,
    ...), not always a circle, so a grouped point batch can mix node
    types: a circle carries its center directly (``cy``), a square
    carries its top-left corner and side length (``y``/``h``)."""
    return node["cy"] if "cy" in node else node["y"] + node["h"] / 2


def _quiet(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build()


#: Every value channel ``Chart.encode`` accepts (the decision's
#: completeness list), each read off the point batch's scene JSON: *field*
#: is the fixture column each case binds that channel to, and *read*
#: extracts the per-row value the rendered scene carries for it. A channel
#: that never reached the points reads as all-``None``/the mark's own
#: default (a single repeated size, ``opacity=1.0`` for every row); one that
#: did varies with the bound column, exactly like a plain chart's.
_NUMERIC_CHANNELS: dict[str, Callable[[dict], list]] = {
    "size": lambda batch: [n["r"] for n in batch["nodes"]],
    "opacity": lambda batch: [n["style"]["opacity"] for n in batch["nodes"]],
    "fill_opacity": lambda batch: [n["style"].get("fill_opacity") for n in batch["nodes"]],
    "stroke_opacity": lambda batch: [n["style"].get("stroke_opacity") for n in batch["nodes"]],
    "stroke_width": lambda batch: [n["style"]["stroke_width"] for n in batch["nodes"]],
}
_STRING_CHANNELS: dict[str, Callable[[dict], list]] = {
    # `.get(..., []) or []`: a dropped/unbound channel reads back `None`
    # (tooltips) or is absent (hrefs/keys/descriptions) rather than raising,
    # so a comparison against the sibling's real values fails on content
    # (a clean, readable diff) instead of crashing on `None` iteration.
    "tooltip": lambda batch: [t["fields"][0]["value"] for t in (batch.get("tooltips") or [])],
    "href": lambda batch: batch.get("hrefs") or [],
    "key": lambda batch: batch.get("keys") or [],
    "description": lambda batch: batch.get("descriptions") or [],
}
_VALUE_CHANNELS: dict[str, tuple[str, Callable[[dict], list]]] = {
    **{name: ("s", read) for name, read in _NUMERIC_CHANNELS.items()},
    **{name: ("id", read) for name, read in _STRING_CHANNELS.items()},
}


def _sibling_values(
    channel: str, field: str, read: Callable[[dict], list], frame: pl.DataFrame
) -> list:
    """What *channel* reads for *field* on a plain chart over the same column.

    The one reference every "reached the points" assertion below compares
    against: a plain point chart's own channel resolution depends only on
    *field*'s own values (and, for size/opacity, its domain), never on the
    fit, so it is the sibling route that proves the residual points draw
    the SAME thing a non-residual chart would, rather than merely "didn't
    raise".
    """
    sibling = fm.Chart(frame).mark_point().encode(x="v", y="w", **{channel: field})
    return read(_point_batch(sibling))


# ---------------------------------------------------------------------------
# The core sweep: every value channel, both forms, both call-time doors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", sorted(_VALUE_CHANNELS))
@pytest.mark.parametrize("door", ["keyword", "encode_dict", "post_hoc"])
@pytest.mark.parametrize("form", ["layered", "flat"])
def test_a_value_channel_on_a_data_column_reaches_the_residual_points(form, door, channel):
    field, read = _VALUE_CHANNELS[channel]
    frame = _frame()
    kwargs = _flat_kwargs() if form == "flat" else {}
    if door == "keyword":
        kwargs[channel] = field
        chart = fm.residplot(frame, x="v", y="w", **kwargs)
    elif door == "encode_dict":
        kwargs["encode"] = {channel: field}
        chart = fm.residplot(frame, x="v", y="w", **kwargs)
    else:
        chart = fm.residplot(frame, x="v", y="w", **kwargs).encode(**{channel: field})
    # The layered form's summary layers warn on this door (they lack the
    # channel at the call, or their own fit grid never carries it): the
    # point layer's own values are this test's one claim either way.
    values = _quiet(lambda: read(_point_batch(chart)))
    assert values == _sibling_values(channel, field, read, frame)


# ---------------------------------------------------------------------------
# The layered form: a keyword/`encode=` value channel binds the point
# layer alone, never the chart, so the summary layers (one mark for the
# whole fit) and the lowess overlay (its own fit grid) never inherit it.
# A LATER `.encode()` binds the whole chart, so every layer inherits it:
# the summary layers and the lowess overlay lack it and drop-and-warn.
# ---------------------------------------------------------------------------


def _layer_style_and_tooltips(batch: dict) -> tuple:
    """*batch*'s per-node style and tooltip fields, geometry left out.

    What a summary or lowess layer draws when nothing it reads changes --
    compared against the same layer built with no value channel bound at
    all, so a value channel's own influence (none, for these layers) is
    the one thing either assertion can catch.
    """
    return (
        [n.get("style") for n in batch["nodes"]],
        batch.get("tooltips"),
    )


@pytest.mark.parametrize("lowess", [False, True])
@pytest.mark.parametrize("door", ["keyword", "encode_dict"])
@pytest.mark.parametrize("channel", ["size", "opacity", "tooltip"])
def test_a_value_channel_bound_at_the_call_never_reaches_the_summary_or_lowess_layers(
    channel, door, lowess
):
    field, read = _VALUE_CHANNELS[channel]
    frame = _frame()
    kwargs: dict = {"lowess": lowess}
    if door == "keyword":
        kwargs[channel] = field
    else:
        kwargs["encode"] = {channel: field}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batches = _scene_marks(fm.residplot(frame, x="v", y="w", **kwargs))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert read(batches[0]) == _sibling_values(channel, field, read, frame)
    unbound_batches = _scene_marks(fm.residplot(frame, x="v", y="w", lowess=lowess))
    # Same layers, in the same order, before the pairwise zip below --
    # otherwise a value channel misread as a grouping channel (dropping
    # the metrics layer) would zip short and pass unnoticed.
    assert [b["kind"] for b in batches] == [b["kind"] for b in unbound_batches]
    for drawn, unbound in zip(batches[1:], unbound_batches[1:]):
        assert _layer_style_and_tooltips(drawn) == _layer_style_and_tooltips(unbound)


#: Each later-door case and the exact warnings its summary/lowess layers
#: give, in layer order (rule, text, and -- under `lowess=True` -- line).
_LATER_VALUE_CHANNEL_CASES: dict[str, tuple[str, bool, list[str]]] = {
    "size": (
        "s",
        True,
        [
            "residplot: size 's' was dropped: the statistic this text mark draws does not "
            "carry 's', so it draws without size",
            "residplot: size 's' was dropped: the statistic this line mark draws does not "
            "carry 's', so it draws without size",
        ],
    ),
    "opacity": (
        "s",
        True,
        [
            "residplot: opacity 's' was dropped: the statistic this rule mark draws does not "
            "carry 's', so it draws without opacity",
            "residplot: opacity 's' was dropped: the statistic this text mark draws does not "
            "carry 's', so it draws without opacity",
            "residplot: opacity 's' was dropped: the statistic this line mark draws does not "
            "carry 's', so it draws without opacity",
        ],
    ),
    "tooltip": (
        "id",
        False,
        [
            "residplot: tooltip 'id' was dropped: the statistic this rule mark draws does not "
            "carry 'id', so it draws without tooltip",
            "residplot: tooltip 'id' was dropped: the statistic this text mark draws does not "
            "carry 'id', so it draws without tooltip",
        ],
    ),
}


@pytest.mark.parametrize("case", sorted(_LATER_VALUE_CHANNEL_CASES))
def test_a_later_value_channel_reaches_the_points_while_the_summary_and_lowess_layers_warn(case):
    channel = case
    field, lowess, expected_warnings = _LATER_VALUE_CHANNEL_CASES[case]
    read = _VALUE_CHANNELS[channel][1]
    frame = _frame()
    reset_warnings()
    chart = fm.residplot(frame, x="v", y="w", lowess=lowess).encode(**{channel: field})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batches = _scene_marks(chart)
    assert read(batches[0]) == _sibling_values(channel, field, read, frame)
    unbound_batches = _scene_marks(fm.residplot(frame, x="v", y="w", lowess=lowess))
    assert [b["kind"] for b in batches] == [b["kind"] for b in unbound_batches]
    for drawn, unbound in zip(batches[1:], unbound_batches[1:]):
        drawn_style, drawn_tooltips = _layer_style_and_tooltips(drawn)
        unbound_style, unbound_tooltips = _layer_style_and_tooltips(unbound)
        assert drawn_style == unbound_style
        if channel == "tooltip":
            # An explicit chart-level tooltip suppresses every layer's own
            # default tooltip fields, not only the ones that end up
            # dropping it (_spec_build.SpecBuildMixin._inject_auto_tooltips's
            # early return), so a summary layer that drops the inherited
            # binding is left with none at all -- the same outcome an
            # existing (non-fit-summary) statistic sub-layer's dropped
            # tooltip already gives, not a fallback to the unbound
            # sibling's own default.
            assert drawn_tooltips is None
        else:
            assert drawn_tooltips == unbound_tooltips
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert messages == expected_warnings


def test_a_flat_post_hoc_value_channel_sizes_the_points():
    frame = _frame()
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs()).encode(size="s")
    values = [n["r"] for n in _point_batch(chart)["nodes"]]
    assert values == _sibling_values("size", "s", _NUMERIC_CHANNELS["size"], frame)


def test_only_the_reference_and_metrics_layers_are_fit_summaries():
    layers = json.loads(fm.residplot(_frame(), x="v", y="w").to_spec().to_json())["layers"]
    fit_summary_by_name = {layer["name"]: layer.get("fit_summary", False) for layer in layers}
    assert fit_summary_by_name == {"point": False, "reference": True, "metrics": True}
    plain_layers = json.loads(
        fm.Chart(_frame()).mark_point().encode(x="v", y="w").layer().to_spec().to_json()
    )["layers"]
    assert all("fit_summary" not in layer for layer in plain_layers)


def test_residplots_summary_warning_reads_as_lmplots_metrics_text():
    """residplot's corner-metrics drop-and-warn is the same sentence
    lmplot's own metrics-text layer already gives for a binding its fit
    grid does not carry, apart from the entry each names."""
    frame = _frame()
    reset_warnings()
    with warnings.catch_warnings(record=True) as lm_caught:
        warnings.simplefilter("always")
        fm.lmplot(frame, x="v", y="w", show_metrics=True).encode(size="s").to_svg()
    with warnings.catch_warnings(record=True) as resid_caught:
        warnings.simplefilter("always")
        fm.residplot(frame, x="v", y="w", zero_line=False).encode(size="s").to_svg()
    lm_text = next(
        str(w.message) for w in lm_caught if "text mark draws does not carry" in str(w.message)
    )
    resid_text = next(
        str(w.message) for w in resid_caught if "text mark draws does not carry" in str(w.message)
    )
    assert lm_text.removeprefix("lmplot: ") == resid_text.removeprefix("residplot: ")


# ---------------------------------------------------------------------------
# Grouping and value channels compose: a grouping keyword still refits per
# group, and an explicit value channel replaces the chart's default
# tooltip list rather than adding to it, the same as on any chart.
# ---------------------------------------------------------------------------


def test_color_grouping_still_refits_per_group_while_an_explicit_tooltip_replaces_the_default_list():
    """An explicit ``tooltip=`` replaces the default tooltip list, the way
    it does on a plain chart and on every regression figure (Vega-Lite's
    tooltip channel replaces the same way); grouping still refits each
    group's own residuals underneath it."""
    frame = _frame(g=["p", "p", "p", "p", "q", "q"])
    chart = fm.residplot(frame, x="v", y="w", color="g", tooltip="id")
    batch = _point_batch(chart)
    # (a) refit per group: two distinct point fills (one per group), and
    # residual positions that differ from a pooled, ungrouped fit over the
    # same rows.
    fills = {
        (n["style"]["fill"]["r"], n["style"]["fill"]["g"], n["style"]["fill"]["b"])
        for n in batch["nodes"]
    }
    assert len(fills) == 2
    pooled_cy = [
        n["cy"] for n in _point_batch(fm.residplot(frame, x="v", y="w", **_flat_kwargs()))["nodes"]
    ]
    grouped_cy = [
        n["cy"]
        for n in _point_batch(fm.residplot(frame, x="v", y="w", color="g", **_flat_kwargs()))[
            "nodes"
        ]
    ]
    assert grouped_cy != pooled_cy
    # (b) the point batch's tooltip field names are exactly the explicit
    # binding's, the same as a plain chart's -- `tooltip=` replaces the
    # grouping field's own default entry rather than listing both.
    names = {f["name"] for t in batch["tooltips"] for f in t["fields"]}
    assert names == {"id"}
    plain = fm.Chart(frame).mark_point().encode(x="v", y="w", color="g", tooltip="id")
    plain_names = {f["name"] for t in _point_batch(plain)["tooltips"] for f in t["fields"]}
    assert names == plain_names


@pytest.mark.parametrize("keyword", ["color", "fill", "detail", "shape", "stroke_dash"])
def test_a_grouping_keyword_still_refits_per_group_and_drops_the_corner_metrics(keyword):
    frame = _frame(g=["p", "p", "p", "p", "q", "q"])
    chart = _quiet(lambda: fm.residplot(frame, x="v", y="w", **{keyword: "g"}))
    batches = _scene_marks(chart)
    # Refit per group means more than one point-row group; the single-fit
    # metrics-text layer is dropped (no single corner for >1 group), so the
    # layered default shrinks from 3 batches (point, rule, text) to 2.
    assert [b["kind"] for b in batches] == ["point", "rule"]
    grouped_cy = [_node_cy(n) for n in batches[0]["nodes"]]
    pooled_cy = [
        _node_cy(n)
        for n in _point_batch(_quiet(lambda: fm.residplot(frame, x="v", y="w")))["nodes"]
    ]
    assert grouped_cy != pooled_cy


def test_the_label_keyword_keeps_one_group_and_the_corner_metrics():
    """``label=`` injects a constant, single-level column (one group by
    construction), unlike the other grouping keywords above, so the corner
    metrics stay drawn rather than dropped (residplot's own documented
    exception)."""
    frame = _frame()
    chart = _quiet(lambda: fm.residplot(frame, x="v", y="w", label="L"))
    batches = _scene_marks(chart)
    assert [b["kind"] for b in batches] == ["point", "rule", "text"]


# ---------------------------------------------------------------------------
# Negative controls: constants, mark overrides, a column the data lacks,
# and the plain-chart rule for `text=` on a point mark.
# ---------------------------------------------------------------------------


#: Each constant-channel case and the exact `Chart.encode` text it raises,
#: on both the keyword and `encode=` doors -- the two doors that bind the
#: point layer alone on the layered form, and so the two a constant could
#: slip past if the layered form's own encode() validation skipped them.
_CONSTANT_VALUE_CHANNEL_CASES: dict[str, tuple[Any, str]] = {
    "size": (30, r"encode\(size=\.\.\.\) expects str, Size instance"),
    "opacity": (0.5, r"encode\(opacity=\.\.\.\) expects str, Opacity instance"),
}


@pytest.mark.parametrize("channel", sorted(_CONSTANT_VALUE_CHANNEL_CASES))
@pytest.mark.parametrize("door", ["keyword", "encode_dict"])
def test_a_constant_value_channel_is_refused_unchanged(channel, door):
    value, match = _CONSTANT_VALUE_CHANNEL_CASES[channel]
    frame = _frame()
    kwargs = {channel: value} if door == "keyword" else {"encode": {channel: value}}
    with pytest.raises(TypeError, match=match):
        fm.residplot(frame, x="v", y="w", **kwargs)


def test_mark_override_size_renders_unchanged():
    frame = _frame()
    chart = fm.residplot(frame, x="v", y="w", mark={"point": {"size": 30}})
    radii = [n["r"] for n in _point_batch(chart)["nodes"]]
    sibling = fm.Chart(frame).mark_point(size=30).encode(x="v", y="w")
    assert radii == [n["r"] for n in _point_batch(sibling)["nodes"]]


def test_layered_tooltip_on_a_missing_column_is_refused_unchanged():
    frame = _frame().drop("id")
    with pytest.raises(ValueError, match=r"unknown column 'zz'"):
        fm.residplot(frame, x="v", y="w", tooltip="zz").to_svg()


def test_flat_tooltip_on_a_missing_column_is_dropped_without_a_word():
    frame = _frame().drop("id")
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), tooltip="zz")
    batch = _point_batch(chart)
    assert batch.get("tooltips") is None


@pytest.mark.parametrize(
    "form,kwargs,tooltip_names",
    [
        ("flat", _flat_kwargs(), {"v", "w residual", "id"}),
        ("layered", {}, {"v", "w residual"}),
    ],
)
def test_text_channel_on_a_point_is_not_drawn(form, kwargs, tooltip_names):
    """``mark_point`` does not draw ``text=``, on either form, so the SVG
    is unchanged -- but the point's default auto-tooltip, computed from
    what the mark actually draws, still differs: flat's one mark reads
    ``text=`` straight off the chart-level encoding and lists it, while
    the layered form's point layer never binds a chart-level ``text=`` at
    all (a value channel binds the point layer alone on that form)."""
    frame = _frame()
    plain = fm.residplot(frame, x="v", y="w", **kwargs).to_svg()
    with_text = fm.residplot(frame, x="v", y="w", **kwargs, text="id").to_svg()
    assert plain == with_text
    names = {
        f["name"]
        for t in _point_batch(fm.residplot(frame, x="v", y="w", **kwargs, text="id"))["tooltips"]
        for f in t["fields"]
    }
    assert names == tooltip_names


# ---------------------------------------------------------------------------
# The caller's own data is never renamed: residplot's computed columns
# take a fresh name instead, when the data already uses one of theirs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collision_column", ["residual", "_ref_zero", "_metrics_text", "_metrics_y", "x"]
)
def test_an_unbound_clashing_column_stays_the_callers_and_renders_byte_identical(collision_column):
    plain = _frame()
    values = ["z0", "z1", "z2", "z3", "z4", "z5"]
    with_clash = _frame(**{collision_column: values})
    chart = fm.residplot(with_clash, x="v", y="w")
    # The caller's own frame is never renamed: the column stays exactly
    # where it was, under its own name, with its own values. The
    # computed column took a fresh name instead -- invisible here, since
    # nothing binds it.
    assert chart._data[collision_column].to_list() == values
    svg_plain = fm.residplot(plain, x="v", y="w").to_svg()
    assert svg_plain == chart.to_svg()
    # The wire itself: the fit's own transform carries a residual_columns
    # setting for the clashing role, naming the fresh column it chose --
    # not merely a render that happens to look right.
    role = collision_column.lstrip("_")
    smooth_spec = json.loads(chart.to_spec().to_json())["transforms"][1]
    assert smooth_spec["residual_columns"][role] == f"{collision_column}_1"
    no_clash_spec = json.loads(fm.residplot(plain, x="v", y="w").to_spec().to_json())["transforms"][
        1
    ]
    assert "residual_columns" not in no_clash_spec


@pytest.mark.parametrize("door", ["keyword", "encode_dict"])
def test_tooltip_residual_reads_the_callers_column_when_the_data_has_its_own(door):
    """The caller's data is never renamed: when it already has a column
    named 'residual', binding tooltip='residual' reads THAT column --
    the computed residual took a fresh name instead."""
    frame = _frame(residual=["r0", "r1", "r2", "r3", "r4", "r5"])
    kwargs = _flat_kwargs()
    if door == "keyword":
        kwargs["tooltip"] = "residual"
    else:
        kwargs["encode"] = {"tooltip": "residual"}
    chart = fm.residplot(frame, x="v", y="w", **kwargs)
    values = _STRING_CHANNELS["tooltip"](_point_batch(chart))
    assert values == _sibling_values("tooltip", "residual", _STRING_CHANNELS["tooltip"], frame)


def test_a_multi_field_tooltip_naming_a_clashing_column_reads_the_callers_column():
    frame = _frame(residual=["r0", "r1", "r2", "r3", "r4", "r5"])
    chart = fm.residplot(
        frame, x="v", y="w", **_flat_kwargs(), tooltip=fm.Tooltip("id", "residual")
    )
    batch = _point_batch(chart)
    fields = [{f["name"]: f["value"] for f in t["fields"]} for t in batch["tooltips"]]
    plain = fm.Chart(frame).mark_point().encode(x="v", y="w", tooltip=fm.Tooltip("id", "residual"))
    plain_fields = [
        {f["name"]: f["value"] for f in t["fields"]} for t in _point_batch(plain)["tooltips"]
    ]
    assert fields == plain_fields


def test_size_on_a_numeric_non_predictor_x_reads_the_callers_column():
    frame = _frame(x=[1.0, 5.0, 10.0, 2.0, 3.0, 4.0])
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), size="x")
    values = [n["r"] for n in _point_batch(chart)["nodes"]]
    assert values == _sibling_values("size", "x", _NUMERIC_CHANNELS["size"], frame)


def test_key_ref_zero_reads_the_callers_column_under_the_default_zero_line():
    frame = _frame(_ref_zero=["a", "b", "c", "d", "e", "f"])
    chart = fm.residplot(frame, x="v", y="w", key="_ref_zero")
    values = _point_batch(chart)["keys"]
    assert values == _sibling_values("key", "_ref_zero", _STRING_CHANNELS["key"], frame)


@pytest.mark.parametrize("door", ["keyword", "encode_dict"])
def test_a_grouping_field_named_like_a_computed_column_groups_by_the_callers_column(door):
    frame = _frame(residual=["p", "p", "p", "q", "q", "q"])
    kwargs = _flat_kwargs()
    if door == "keyword":
        kwargs["color"] = "residual"
    else:
        kwargs["encode"] = {"color": "residual"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = fm.residplot(frame, x="v", y="w", **kwargs)
        batch = _point_batch(chart)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    fills = {
        (n["style"]["fill"]["r"], n["style"]["fill"]["g"], n["style"]["fill"]["b"])
        for n in batch["nodes"]
    }
    assert len(fills) == 2
    grouped_cy = [n["cy"] for n in batch["nodes"]]
    pooled_cy = [
        n["cy"] for n in _point_batch(fm.residplot(frame, x="v", y="w", **_flat_kwargs()))["nodes"]
    ]
    assert grouped_cy != pooled_cy
    # The wire: groupby names the caller's own 'residual' column, and the
    # fit's own residual output still took a fresh name beside it.
    smooth_spec = json.loads(chart.to_spec().to_json())["transforms"][1]
    assert smooth_spec["groupby"] == ["residual"]
    assert smooth_spec["residual_columns"]["residual"] == "residual_1"


def test_shape_on_a_clashing_column_groups_by_the_callers_column():
    frame = _frame(x=["p", "p", "p", "q", "q", "q"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), shape="x")
        grouped_cy = [_node_cy(n) for n in _point_batch(chart)["nodes"]]
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    pooled_cy = [
        _node_cy(n)
        for n in _point_batch(fm.residplot(frame, x="v", y="w", **_flat_kwargs()))["nodes"]
    ]
    assert grouped_cy != pooled_cy


@pytest.mark.parametrize("door", ["keyword", "encode_dict"])
def test_a_facet_at_the_call_on_a_clashing_column_facets_by_the_callers_column(door):
    frame = _frame(x=["p", "p", "p", "q", "q", "q"])
    kwargs = _flat_kwargs()
    if door == "keyword":
        kwargs["facet_col"] = "x"
    else:
        kwargs["encode"] = {"facet_col": "x"}
    chart = fm.residplot(frame, x="v", y="w", **kwargs)
    scene_json, _packed = _render_scene(chart)
    panels = json.loads(scene_json)["panels"]
    assert len(panels) == 2


def _panel_points(chart: Any) -> list[list[tuple]]:
    """Each panel's point layer, as a list of ``(cx, cy)`` pairs, in panel order."""
    scene_json, _packed = _render_scene(chart)
    panels = json.loads(scene_json)["panels"]
    return [[(n["cx"], n["cy"]) for n in panel["marks"][0]["nodes"]] for panel in panels]


@pytest.mark.parametrize("form", ["layered", "flat"])
def test_a_facet_method_call_on_a_clashing_column_facets_by_the_callers_column(form):
    """``.facet(col=...)``, a chained call after residplot returns (unlike
    the at-call ``facet_col=`` keyword above), reads the caller's own
    clashing column too, on both forms -- each panel's points equal the
    sibling faceted by the same values under a column name the computed
    predictor never takes, not merely the same panel count (which a wrong
    split of the caller's own rows could still give)."""
    frame = _frame(x=["p", "p", "p", "q", "q", "q"])
    kwargs = _flat_kwargs() if form == "flat" else {}
    chart = fm.residplot(frame, x="v", y="w", **kwargs).facet(col="x")
    points = _panel_points(chart)
    assert len(points) == 2
    sibling = fm.residplot(frame.rename({"x": "grp"}), x="v", y="w", **kwargs).facet(col="grp")
    assert points == _panel_points(sibling)


#: Each later-door clash case: a binding written AFTER residplot returns
#: (not at the call), naming a column that clashes with one of
#: residplot's own computed names. The caller's data is never renamed, so
#: a later door reads exactly what an at-call door does.
@pytest.mark.parametrize(
    "case",
    [
        "tooltip-layered",
        "tooltip-flat",
        "size-x",
        "color-residual",
        "color-x",
        "facet-method",
        "facet-encode",
    ],
)
def test_a_later_binding_naming_a_clashing_column_reads_the_callers_column(case):
    if case.startswith("tooltip"):
        form = case.removeprefix("tooltip-")
        frame = _frame(residual=["r0", "r1", "r2", "r3", "r4", "r5"])
        kwargs = _flat_kwargs() if form == "flat" else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chart = fm.residplot(frame, x="v", y="w", **kwargs).encode(tooltip="residual")
            values = _STRING_CHANNELS["tooltip"](_point_batch(chart))
        assert values == _sibling_values("tooltip", "residual", _STRING_CHANNELS["tooltip"], frame)
    elif case == "size-x":
        frame = _frame(x=[1.0, 5.0, 10.0, 2.0, 3.0, 4.0])
        chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs()).encode(size="x")
        values = [n["r"] for n in _point_batch(chart)["nodes"]]
        assert values == _sibling_values("size", "x", _NUMERIC_CHANNELS["size"], frame)
    elif case in ("color-residual", "color-x"):
        clash_column = case.removeprefix("color-")
        frame = _frame(**{clash_column: ["p", "p", "p", "q", "q", "q"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs()).encode(color=clash_column)
            batch = _point_batch(chart)
        assert not [w for w in caught if issubclass(w.category, UserWarning)]
        fills = {
            (n["style"]["fill"]["r"], n["style"]["fill"]["g"], n["style"]["fill"]["b"])
            for n in batch["nodes"]
        }
        assert len(fills) == 2
        grouped_cy = [n["cy"] for n in batch["nodes"]]
        pooled_cy = [
            n["cy"]
            for n in _point_batch(fm.residplot(frame, x="v", y="w", **_flat_kwargs()))["nodes"]
        ]
        assert grouped_cy != pooled_cy
    else:
        frame = _frame(residual=["p", "p", "p", "q", "q", "q"])
        chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs())
        chart = (
            chart.facet(col="residual")
            if case == "facet-method"
            else chart.encode(facet_col="residual")
        )
        scene_json, _packed = _render_scene(chart)
        panels = json.loads(scene_json)["panels"]
        assert len(panels) == 2


@pytest.mark.parametrize("column", ["_ref_zero", "_metrics_text"])
def test_an_optional_column_this_call_does_not_compute_stays_the_datas_own(column):
    """``zero_line=False``/``show_metrics=False`` means this call's
    transform never emits ``_ref_zero``/``_metrics_text`` at all, so a
    data column by that name is never a reserved name to begin with --
    no fresh name is chosen for anything, and the column is carried like
    any other."""
    frame = _frame(**{column: ["a", "b", "c", "d", "e", "f"]})
    chart = fm.residplot(frame, x="v", y="w", zero_line=False, show_metrics=False, tooltip=column)
    values = _STRING_CHANNELS["tooltip"](_point_batch(chart))
    assert values == _sibling_values("tooltip", column, _STRING_CHANNELS["tooltip"], frame)
    # The wire: no fresh name was chosen for anything -- this call never
    # emits the optional column at all, so it was never a reserved name.
    smooth_spec = json.loads(chart.to_spec().to_json())["transforms"][1]
    assert "residual_columns" not in smooth_spec


def test_tooltip_residual_shows_the_computed_residual_with_no_data_column():
    frame = _frame()
    assert "residual" not in frame.columns
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), tooltip="residual")
    values = _STRING_CHANNELS["tooltip"](_point_batch(chart))
    sibling = (
        fm.Chart(frame)
        .transform(fm.Smooth(x="v", y="w", method="lm", degree=1, ci=None, output="residuals"))
        .mark_point()
        .encode(x="x", y="residual", tooltip="residual")
    )
    assert values == _STRING_CHANNELS["tooltip"](_point_batch(sibling))


@pytest.mark.parametrize("form", ["layered", "flat", "lowess"])
def test_a_predictor_named_x_is_byte_identical_and_not_carried_twice(form):
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 3.0, 1.0, 4.0, 3.5, 5.0]})
    kwargs = {"flat": _flat_kwargs(), "lowess": {"lowess": True}, "layered": {}}[form]
    svg = fm.residplot(frame, x="x", y="y", **kwargs).to_svg()
    # A predictor renamed to a column the computed `x` never collides
    # with renders exactly the same, re-titled back to `x`: proof this is
    # the predictor's own values, not a second, carried copy of them.
    renamed = frame.rename({"x": "p"})
    sibling = fm.residplot(renamed, x=fm.X("p", title="x"), y="y", **kwargs).to_svg()
    assert svg == sibling


# ---------------------------------------------------------------------------
# No value channel bound: a harmless unrelated extra column changes nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"zero_line": False, "show_metrics": False},
        {"lowess": True},
        {"robust": True},
        {"label": "L"},
        {"color": "g"},
    ],
    ids=["layered-default", "flat", "lowess", "robust", "label", "color-grouped"],
)
def test_every_residplot_with_no_value_channel_is_byte_identical(kwargs):
    base = {
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "w": [2.0, 3.0, 1.0, 4.0, 3.5, 5.0, 2.5, 1.5],
        "g": ["p", "p", "p", "p", "q", "q", "q", "q"],
    }
    plain = pl.DataFrame(base)
    with_extra = pl.DataFrame({**base, "extra": [10, 20, 30, 40, 50, 60, 70, 80]})
    svg_plain = _quiet(lambda: fm.residplot(plain, x="v", y="w", **kwargs).to_svg())
    svg_with_extra = _quiet(lambda: fm.residplot(with_extra, x="v", y="w", **kwargs).to_svg())
    assert svg_plain == svg_with_extra


# ---------------------------------------------------------------------------
# A log x: the off-side warning is unaffected by the carried columns.
# ---------------------------------------------------------------------------


def test_residplot_on_a_log_x_warns_once_and_carries_a_value_channel():
    reset_warnings()
    frame = pl.DataFrame(
        {"v": [-1.0, 0.0, 1.0, 2.0, 4.0, 8.0], "id": ["a", "b", "c", "d", "e", "f"]}
    )
    chart = fm.residplot(
        frame, x=fm.X("v", scale=fm.LogScale()), y="v", **_flat_kwargs(), tooltip="id"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = _point_batch(chart)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1, messages
    assert messages[0] == (
        "x scale: 2 row(s) of 'v' are zero or negative, which a LogScale "
        "cannot place, so they are not drawn; filter them, or use "
        "fm.SymlogScale() for data that reaches zero"
    )
    # The 4 on-side rows carry their own `id` through unaffected by the
    # log remapping (pass-through columns are copied, never log-mapped).
    on_side_ids = _STRING_CHANNELS["tooltip"](batch)
    assert set(on_side_ids) == {"c", "d", "e", "f"}


# ---------------------------------------------------------------------------
# Degenerate data shapes: zero rows and one row keep their existing
# degenerate-fit path, with a value channel resolving rather than refusing.
# ---------------------------------------------------------------------------


def test_zero_row_data_resolves_a_value_channel_without_a_new_refusal():
    reset_warnings()
    frame = pl.DataFrame(
        {
            "v": pl.Series([], dtype=pl.Float64),
            "w": pl.Series([], dtype=pl.Float64),
            "s": pl.Series([], dtype=pl.Float64),
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.residplot(frame, x="v", y="w", **_flat_kwargs(), size="s").to_svg()
    messages = {str(w.message) for w in caught if issubclass(w.category, UserWarning)}
    assert messages == {
        "Smooth: could not fit the data (fewer than 2 usable (non-null/non-NaN) "
        "x/y points); it degraded to an empty/flat output instead of failing "
        "the whole chart",
        "the chart has no rows to draw after its transforms; its panel is drawn empty",
    }


def test_one_row_data_keeps_the_degenerate_fit_path_with_a_value_channel():
    reset_warnings()
    frame = pl.DataFrame({"v": [1.0], "w": [2.0], "s": [3.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.residplot(frame, x="v", y="w", **_flat_kwargs(), size="s").to_svg()
    messages = {str(w.message) for w in caught if issubclass(w.category, UserWarning)}
    assert messages == {
        "Smooth: could not fit the data (fewer than 2 usable (non-null/non-NaN) "
        "x/y points); it degraded to an empty/flat output instead of failing "
        "the whole chart",
        "the chart has no rows to draw after its transforms; its panel is drawn empty",
    }


# ---------------------------------------------------------------------------
# Null handling and dtype fidelity in a carried column.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,tag",
    [
        ("null", ["a", None, "c", None, "e", "f"]),
        ("nan", [1.0, float("nan"), 3.0, float("nan"), 5.0, 6.0]),
    ],
)
def test_a_null_or_nan_in_a_carried_column_is_copied_as_the_callers_row_held_it(case, tag):
    frame = _frame(tag=tag)
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), tooltip="tag")
    values = _STRING_CHANNELS["tooltip"](_point_batch(chart))
    assert values == _sibling_values("tooltip", "tag", _STRING_CHANNELS["tooltip"], frame)


@pytest.mark.parametrize(
    "column,dtype,values",
    [
        ("utf8_col", pl.Utf8, ["a", "b", "a", "b", "a", "b"]),
        ("int_col", pl.Int64, [1, 2, 3, 4, 5, 6]),
        ("bool_col", pl.Boolean, [True, False, True, False, True, False]),
        ("cat_col", pl.Categorical, ["a", "b", "a", "b", "a", "b"]),
    ],
)
def test_a_carried_column_of_every_dtype_is_copied_unchanged(column, dtype, values):
    frame = _frame(**{column: pl.Series(values, dtype=dtype)})
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), key=column)
    carried = _point_batch(chart)["keys"]
    assert carried == _sibling_values("key", column, _STRING_CHANNELS["key"], frame)


def test_a_carried_datetime_column_is_copied_unchanged():
    import datetime

    frame = _frame(
        dt_col=[datetime.datetime(2020, 1, i + 1) for i in range(6)],
    )
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs(), key="dt_col")
    values = _point_batch(chart)["keys"]
    assert values == _sibling_values("key", "dt_col", _STRING_CHANNELS["key"], frame)
    assert len(set(values)) == 6


# ---------------------------------------------------------------------------
# `size=None` is unset, not a bound constant.
# ---------------------------------------------------------------------------


def test_size_none_is_byte_identical():
    frame = _frame()
    without = fm.residplot(frame, x="v", y="w").to_svg()
    with_none = fm.residplot(frame, x="v", y="w", size=None).to_svg()
    assert without == with_none


# ---------------------------------------------------------------------------
# Every door that reads the caller's rows directly -- a `.transform()`
# step, an attached layer -- reads a column clashing with one of
# residplot's computed names under the caller's own name, the data's.
# ---------------------------------------------------------------------------


def test_a_transform_step_reads_a_clashing_column_under_the_callers_name():
    frame = _frame(residual=[50.0, 60.0, 150.0, 70.0, 80.0, 90.0])
    # `> 55` keeps 5 of the 6 rows -- several, not a near-empty panel --
    # so a filter that read the COMPUTED residual instead (every one of
    # which differs from these literal values) would keep a different
    # row count, not merely degenerate to the same empty render either
    # way would give under `> 100`.
    chart = fm.residplot(frame, x="v", y="w", **_flat_kwargs()).transform(
        fm.transform_filter("datum.residual > 55")
    )
    svg = chart.to_svg()
    sibling = fm.residplot(
        frame.filter(pl.col("residual") > 55), x="v", y="w", **_flat_kwargs()
    ).to_svg()
    assert svg == sibling


def test_an_attached_layer_reads_a_clashing_column_under_the_callers_name():
    frame = _frame(residual=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    overlay = fm.Layer(mark="point", encoding={"x": "v", "y": "residual"})
    via_layer = fm.residplot(frame, x="v", y="w", **_flat_kwargs()).layer(overlay).to_svg()
    via_layers_kwarg = fm.residplot(
        frame, x="v", y="w", **_flat_kwargs(), layers=[overlay]
    ).to_svg()
    assert via_layer == via_layers_kwarg
    # A sibling with no clash at all -- the same values under a name the
    # computed residual never takes -- renders byte-identically if the
    # layer above really read the caller's own 'residual' column: reading
    # the COMPUTED residual instead would draw different y positions,
    # since this frame's own 'residual' values are not the fit's.
    no_clash = frame.rename({"residual": "resid_data"})
    sibling_overlay = fm.Layer(mark="point", encoding={"x": "v", "y": "resid_data"})
    sibling = fm.residplot(no_clash, x="v", y="w", **_flat_kwargs()).layer(sibling_overlay).to_svg()
    assert via_layer == sibling


# ---------------------------------------------------------------------------
# The residual-column setting is internal: it is not reachable on a
# fitted spec, and it never becomes a public fm.Smooth/fm.Robust kwarg.
# ---------------------------------------------------------------------------


def test_the_residual_column_setting_is_internal_and_residual_mode_only():
    with pytest.raises(ValueError) as smooth_err:
        fm.Smooth(x="v", y="w", output="fitted")._with_residual_columns(residual="r")
    assert str(smooth_err.value) == "Smooth: residual column names apply only to output='residuals'"
    with pytest.raises(ValueError) as robust_err:
        fm.Robust(x="v", y="w")._with_residual_columns(residual="r")
    assert str(robust_err.value) == "Robust: residual column names apply only to output='residuals'"
    assert "residual_columns" not in inspect.signature(fm.Smooth).parameters
    assert "residual_columns" not in inspect.signature(fm.Robust).parameters
    # Direct use (no residplot, no residual_columns setting at all) still
    # collides on the data's own 'residual' -- residplot's naming
    # mechanism is additive to this path, not a change to it.
    frame = _frame(residual=["z0", "z1", "z2", "z3", "z4", "z5"])
    with pytest.raises(ValueError) as collision_err:
        fm.Chart(frame).transform(
            fm.Smooth(x="v", y="w", method="lm", degree=1, ci=None, output="residuals")
        ).to_svg()
    assert str(collision_err.value) == (
        "transform failed: ValueError: stat_smooth: residual output column 'residual' "
        "collides with an input column of the same name; rename the input column"
    )
    # The setting is internal: it changes nothing a caller can read back
    # off the spec through repr().
    smooth = fm.Smooth(x="v", y="w", method="lm", degree=1, ci=None, output="residuals")
    before = repr(smooth)
    assert repr(smooth._with_residual_columns(x="x_1")) == before
