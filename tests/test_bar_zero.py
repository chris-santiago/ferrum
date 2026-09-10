"""Bar chart y-axis zero-anchoring default."""

import json

import polars as pl
import pytest

import ferrum as fm


def _enc(chart, channel):
    """Return the encoding dict for `channel` from chart.to_spec().to_json()."""
    d = json.loads(chart.to_spec().to_json())
    return d.get("encoding", {}).get(channel, {})


# ---------------------------------------------------------------------------
# Gate-interaction regression (batch-C task 4 quality-review remediation,
# S4): the zero-anchor injection at ``_spec_build.py`` used to stamp
# ``"zero": True`` onto whatever scale dict a bar's y-channel already
# carried, REGARDLESS of that scale's ``type``. Before the raw-dict scale
# key gate (F-L04-07) landed, this was a harmless no-op for any non-linear
# type — ``zero`` isn't a field on ``ScaleSpec::Log``/``Symlog``/``Sqrt``/
# ``Pow``/``Time``/``Band``/etc., so serde's flatten carve-out silently
# dropped it and the chart rendered. The gate now refuses an unrecognized
# key outright, so every one of these previously-silent-no-op combinations
# started raising ``ValueError: ... unknown key 'zero' ...`` instead of
# rendering — a real regression the gate's own test suite could not see
# (``tests/test_bar_zero.py`` only ever exercised linear y). This doubles as
# the S4 regression suite the quality reviewers asked for: every case here
# is a documented public spelling (``docs/site/guide/marks-encodings.md``,
# ``docs/site/guide/concepts/secondary-axes.md``) that must keep rendering.
# ---------------------------------------------------------------------------

_NON_LINEAR_Y_SCALE_CASES = [
    ("class:LogScale", lambda: fm.LogScale()),
    ("class:SymlogScale", lambda: fm.SymlogScale()),
    ("class:SqrtScale", lambda: fm.SqrtScale()),
    ("dict:log", lambda: {"type": "log"}),
    ("dict:time", lambda: {"type": "time"}),
    ("dict:pow", lambda: {"type": "pow"}),
]
_NON_LINEAR_Y_SCALE_IDS = [name for name, _factory in _NON_LINEAR_Y_SCALE_CASES]


@pytest.mark.parametrize(
    "name, scale_factory", _NON_LINEAR_Y_SCALE_CASES, ids=_NON_LINEAR_Y_SCALE_IDS
)
def test_bar_non_linear_y_scale_renders_without_zero_key(name, scale_factory):
    """A bar with a non-linear, non-band y-scale (class or raw-dict spelling)
    renders — the zero-anchor injection must not stamp a ``zero`` key onto a
    scale type that has no such field.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale_factory()))
    scale = _enc(chart, "y").get("scale", {})
    assert "zero" not in scale, (
        f"{name}: zero-anchor must not inject a key {scale['type']!r} has no field for"
    )
    svg = chart.to_svg()  # must not raise
    assert "<svg" in svg


def test_bar_log_scale_class_and_dict_spellings_render_byte_identical():
    """``fm.LogScale()`` and ``{"type": "log"}`` reach the identical resolved
    scale — the class path isn't a distinct escape hatch from the dict path.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    svg_class = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LogScale())).to_svg()
    svg_dict = (
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale={"type": "log"})).to_svg()
    )
    assert svg_class == svg_dict


def test_bar_band_scale_with_padding_renders_with_padding_honored():
    """A horizontal bar's y-scale explicitly typed ``band`` (class or dict,
    with ``padding_inner`` set) renders and the padding value survives —
    the sharpest repro from the quality review: no raw dict at all is
    needed, just ``fm.BandScale(padding_inner=...)`` on an hbar's y.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart_class = (
        fm.Chart(df)
        .mark_bar(orient="horizontal")
        .encode(x="val", y=fm.Y("cat", scale=fm.BandScale(padding_inner=0.2)))
    )
    chart_dict = (
        fm.Chart(df)
        .mark_bar(orient="horizontal")
        .encode(x="val", y=fm.Y("cat", scale={"type": "band", "paddingInner": 0.2}))
    )
    scale_class = _enc(chart_class, "y").get("scale", {})
    scale_dict = _enc(chart_dict, "y").get("scale", {})
    assert "zero" not in scale_class
    assert "zero" not in scale_dict
    assert scale_class.get("paddingInner") == pytest.approx(0.2)
    assert scale_dict.get("paddingInner") == pytest.approx(0.2)
    assert chart_class.to_svg() == chart_dict.to_svg()


def test_bar_override_y_scale_type_log_renders():
    """``.override(y_scale_type="log")`` on a plain bar (no explicit scale=)
    renders — the sharper composition-order bug: the zero-anchor injection
    used to run BEFORE the override merge, so a plain-linear zero-anchored
    scale (``{"type": "linear", "zero": True}``) got ``type`` overwritten to
    ``"log"`` by the override AFTER zero was already stamped on, producing
    the gate-refused ``{"type": "log", "zero": True}``. The zero-anchor
    injection now runs AFTER the override merge so it sees the final,
    fully-resolved effective type.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y="val").override(y_scale_type="log")
    scale = _enc(chart, "y").get("scale", {})
    assert scale.get("type") == "log"
    assert "zero" not in scale
    svg = chart.to_svg()  # must not raise
    assert "<svg" in svg


def test_bar_default_zero_anchor():
    """mark_bar injects scale.zero=True on y-encoding by default."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y="val")
    scale = _enc(chart, "y").get("scale", {})
    assert scale.get("zero") is True


def test_bar_explicit_domain_no_zero():
    """User-supplied domain suppresses the zero injection."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale={"domain": [5, 25]}))
    scale = _enc(chart, "y").get("scale", {})
    assert "zero" not in scale or scale.get("zero") is not True
    assert scale.get("domain") == [5, 25]


def test_bar_explicit_zero_false():
    """User can opt out of zero-anchoring."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale={"zero": False}))
    scale = _enc(chart, "y").get("scale", {})
    assert scale.get("zero") is False


def test_non_bar_mark_no_zero_injection():
    """Non-bar marks do not get the zero injection on y."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 15]})
    chart = fm.Chart(df).mark_line().encode(x="x", y="y")
    scale = _enc(chart, "y").get("scale", {})
    assert "zero" not in scale


# ---------------------------------------------------------------------------
# Merged-flat layer path: the zero=False opt-out must survive the
# chart-to-layer conversion (batch F, T5r rider).
#
# `resolve={"y": "independent"}` (and `SecondaryY`, and every
# `.interactive()` LayerChart) renders through the merged single-panel
# `spec.layers` path rather than the composite overlay tree. There, the
# chart-level `scale.zero` injection above never runs -- the renderer's own
# layered-bar rule anchors instead -- so a bar the caller opted out of was
# anchored anyway until `_expand_layers` began forwarding the opt-out as the
# layer's `mark_style.zero`. The pair below is the discriminating shape: the
# opt-out case and the default case must NOT agree.
# ---------------------------------------------------------------------------

_MERGED_FLAT_DF = pl.DataFrame(
    {"cat": ["a", "b", "c"], "val": [60.0, 70.0, 80.0], "other": [55.0, 65.0, 75.0]}
)


def _merged_flat_y_ticks(bar_chart):
    """Render `bar_chart` overlaid with a line under independent-y and return
    the numeric y-axis tick labels."""
    import re

    line = fm.Chart(_MERGED_FLAT_DF).mark_line().encode(x="cat", y="other")
    svg = fm.LayerChart(bar_chart, line, resolve={"y": "independent"}).to_svg()
    return [
        float(t)
        for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        if re.fullmatch(r"-?\d+(\.\d+)?", t or "")
    ]


def test_merged_flat_bar_zero_false_keeps_the_axis_off_zero():
    """`mark_bar(zero=False)` composed into a merged-flat layer stays opted out."""
    bar = fm.Chart(_MERGED_FLAT_DF).mark_bar(zero=False).encode(x="cat", y="val")
    ticks = _merged_flat_y_ticks(bar)
    assert min(ticks) > 0.0, (
        f"zero=False must survive the chart-to-layer conversion; axis ran to {min(ticks)}"
    )


def test_merged_flat_bar_default_still_anchors_at_zero():
    """The default bar keeps the renderer's layered zero-anchor on the same path."""
    bar = fm.Chart(_MERGED_FLAT_DF).mark_bar().encode(x="cat", y="val")
    assert min(_merged_flat_y_ticks(bar)) == 0.0


def test_merged_flat_zero_false_reaches_the_layer_mark_style():
    """The opt-out is carried by the layer's own `mark_style`, which is the
    field the renderer reads (a chart-level `scale.zero` never reaches this
    path)."""
    bar = fm.Chart(_MERGED_FLAT_DF).mark_bar(zero=False).encode(x="cat", y="val")
    line = fm.Chart(_MERGED_FLAT_DF).mark_line().encode(x="cat", y="other")
    merged = fm.LayerChart(bar, line, resolve={"y": "independent"})._build_merged()
    layers = json.loads(merged.to_spec().to_json())["layers"]
    assert layers[0]["mark_style"]["zero"] is False


def test_merged_flat_default_bar_emits_no_zero_key():
    """Charts that leave `zero` at its default serialize as they always did."""
    bar = fm.Chart(_MERGED_FLAT_DF).mark_bar().encode(x="cat", y="val")
    line = fm.Chart(_MERGED_FLAT_DF).mark_line().encode(x="cat", y="other")
    merged = fm.LayerChart(bar, line, resolve={"y": "independent"})._build_merged()
    layers = json.loads(merged.to_spec().to_json())["layers"]
    assert all("zero" not in (ly.get("mark_style") or {}) for ly in layers)
