"""Migration-compat regression tests: B1, B3, B4, F16, F17, M5.

B1  – type_ kwarg alias on channel constructors
B3  – aggregate shorthand auto-infers groupby from sibling non-aggregate fields
B4  – nice=True on Scale objects serializes to {"nice": true} in spec dict
F16 – reverse= on positional scales serializes to {"reverse": true} in spec dict
F17 – Axis(label_map=...) remaps categorical tick labels at render time
M5  – annotate_abline(slope, intercept) draws a slope+intercept line
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# B1: type_ kwarg alias
# ---------------------------------------------------------------------------


def test_type_underscore_alias_stored_as_type():
    """type_= kwarg is normalised to type= internally, not silently dropped."""
    ch = fm.X("hp", type_="Q")
    # _kwargs must store it under the canonical key "type"
    assert ch._kwargs.get("type") == "Q", "type_ was not normalised to type in _kwargs"


def test_type_underscore_alias_in_spec_dict():
    """type_ flows through to_encoding_spec_dict as type_."""
    ch = fm.X("hp", type_="Q")
    spec = ch.to_encoding_spec_dict()
    assert spec.get("type_") == "Q", "type_ kwarg was not present in encoding spec dict"


def test_type_underscore_and_type_are_equivalent():
    """fm.X('hp', type_='Q') and fm.X('hp', type='Q') produce the same spec."""
    ch_underscore = fm.X("hp", type_="Q")
    ch_plain = fm.X("hp", type="Q")
    assert ch_underscore.to_encoding_spec_dict() == ch_plain.to_encoding_spec_dict()


def test_type_underscore_validation_fires():
    """type_ with an invalid value raises ValueError (same as type=)."""
    with pytest.raises(ValueError, match=r"X: type must be one of"):
        fm.X("hp", type_="BAD")


# ---------------------------------------------------------------------------
# B3: aggregate shorthand auto-infers groupby
# ---------------------------------------------------------------------------


def test_aggregate_auto_groupby_renders():
    """encode(x='cat:N', y='mean(val):Q') does not crash and renders bars."""
    df = pl.DataFrame({"cat": ["a", "b", "a", "b"], "val": [1.0, 2.0, 3.0, 4.0]})
    svg = fm.Chart(df).mark_bar().encode(x="cat:N", y="mean(val):Q").to_svg()
    assert "<rect" in svg, "No <rect> elements — bars were not rendered"


def test_aggregate_auto_groupby_multiple_dims():
    """Groupby inference works when multiple non-aggregate channels are present."""
    df = pl.DataFrame(
        {
            "cat": ["a", "b", "a", "b", "a", "b"],
            "grp": ["x", "x", "y", "y", "x", "x"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # Two non-aggregate fields → groupby should be ['cat', 'grp']
    svg = fm.Chart(df).mark_bar().encode(x="cat:N", y="mean(val):Q", color="grp:N").to_svg()
    assert "<rect" in svg


def test_aggregate_auto_groupby_count_shorthand():
    """encode(x='cat:N', y='count():Q') does not crash."""
    df = pl.DataFrame({"cat": ["a", "b", "a", "b", "a"]})
    svg = fm.Chart(df).mark_bar().encode(x="cat:N", y="count():Q").to_svg()
    assert "<rect" in svg


def test_explicit_groupby_not_overridden():
    """An explicit Aggregate(..., groupby=[...]) transform is not altered."""
    df = pl.DataFrame({"cat": ["a", "b", "a", "b"], "val": [1.0, 2.0, 3.0, 4.0]})
    # Explicit transform — should use groupby=['cat'] as specified
    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat:N", y="val:Q")
        .transform(fm.Aggregate([fm.AggregateOp("val", "mean", "val")], groupby=["cat"]))
        .to_svg()
    )
    assert "<rect" in svg


# ---------------------------------------------------------------------------
# B4: nice= on scale objects serializes
# ---------------------------------------------------------------------------


def test_nice_serialized_linear_scale_via_dict():
    """A dict scale with nice=True round-trips through _scale_to_dict unchanged."""
    from ferrum.encoding._scale import _scale_to_dict

    d = _scale_to_dict({"type": "linear", "nice": True})
    assert d.get("nice") is True, "nice was dropped from dict scale"


def test_nice_false_not_forced_into_spec():
    """A dict scale without nice does not gain a spurious nice key."""
    from ferrum.encoding._scale import _scale_to_dict

    d = _scale_to_dict({"type": "linear", "domain": [0, 100]})
    assert "nice" not in d or d["nice"] is False


def test_nice_serialized_typed_linear_scale():
    """LinearScale(nice=True) emits nice: True in the spec dict (#153, FIXED).

    Previously `LinearScale`'s Rust struct carried no `nice` field at all:
    `to_scale_spec` hardcoded `nice: false` regardless of the constructor
    kwarg, so the typed class silently disagreed with the raw-dict
    equivalent (`{"type": "linear", "nice": True}`, which `_scale_to_dict`
    already serialized correctly — see `test_nice_serialized_linear_scale_via_dict`
    above). The class now carries the constructor's `nice=` kwarg onto the
    wire unconditionally (`crates/ferrum-core/src/scale/linear.rs`'s new
    `nice` field), so this asserts the fix rather than documenting the gap.
    """
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import LinearScale

    s = LinearScale(domain=[0, 100], range=[0, 600], nice=True)
    d = _scale_to_dict(s)
    assert d.get("type") == "linear"
    assert d.get("nice") is True, "nice=True on a typed LinearScale must reach the wire spec dict"


def test_nice_serialized_typed_linear_scale_no_domain():
    """The data-derived-domain half of #153: LinearScale(nice=True) with NO
    domain= must ALSO emit nice: True. This is the half that previously
    could not even be eagerly baked into the domain (there is no domain yet
    at construction time), so before the fix it was silently dropped with
    no fallback at all.
    """
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import LinearScale

    s = LinearScale(nice=True)
    d = _scale_to_dict(s)
    assert d.get("type") == "linear"
    assert d.get("nice") is True, "nice=True with no domain must still reach the wire spec dict"


def test_nice_serialized_typed_time_scale():
    """#153 also covers TimeScale (Time and Utc wire arms both)."""
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import TimeScale

    d_time = _scale_to_dict(TimeScale(nice=True))
    assert d_time.get("type") == "time"
    assert d_time.get("nice") is True, (
        "nice=True on a typed TimeScale must reach the wire spec dict"
    )

    d_utc = _scale_to_dict(TimeScale(nice=True, utc=True))
    assert d_utc.get("type") == "utc"
    assert d_utc.get("nice") is True, (
        "nice=True on a typed TimeScale(utc=True) must reach the wire spec dict"
    )


def test_nice_serialized_typed_log_scale():
    """#153 also covers LogScale."""
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import LogScale

    d = _scale_to_dict(LogScale(nice=True))
    assert d.get("type") == "log"
    assert d.get("nice") is True, "nice=True on a typed LogScale must reach the wire spec dict"


def test_nice_serialized_typed_symlog_scale():
    """#153 also covers SymlogScale."""
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import SymlogScale

    d = _scale_to_dict(SymlogScale(nice=True))
    assert d.get("type") == "symlog"
    assert d.get("nice") is True, "nice=True on a typed SymlogScale must reach the wire spec dict"


def test_nice_renders_niced_ticks_for_typed_linear_scale_with_no_domain():
    """#153's disclosed behavior change, pinned at render time: a typed
    LinearScale(nice=True) with no domain now renders the SAME niced ticks
    its raw-dict equivalent (`scale={"type": "linear", "nice": True}`)
    already rendered — before the fix, the typed class silently rendered
    the un-niced ticks instead (data domain [0.5, 9.5] renders ticks 0..10
    niced; the pre-fix typed class instead rendered whatever the encoded
    column's raw min/max happened to be).
    """
    from ferrum._core import LinearScale
    from tests._svg_extents import axis_tick_labels

    df = pl.DataFrame({"x": [0.5, 5.0, 9.5], "y": [1.0, 2.0, 3.0]})

    svg_typed = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=LinearScale(nice=True)), y="y").to_svg()
    )
    svg_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "linear", "nice": True}), y="y")
        .to_svg()
    )
    svg_nonice = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=LinearScale(nice=False)), y="y").to_svg()
    )
    typed_ticks = axis_tick_labels(svg_typed, axis="x")
    dict_ticks = axis_tick_labels(svg_dict, axis="x")
    nonice_ticks = axis_tick_labels(svg_nonice, axis="x")
    assert typed_ticks == dict_ticks, (
        f"typed LinearScale(nice=True) ticks {typed_ticks} must match the raw-dict "
        f"equivalent's ticks {dict_ticks}"
    )
    assert typed_ticks != nonice_ticks, (
        f"nice=True ticks {typed_ticks} must differ from the nice=False baseline {nonice_ticks} "
        f"— otherwise nice= isn't actually reaching the render path"
    )
    # The discriminating signal: nice() WIDENS [0.5, 9.5] outward to exactly
    # [0, 10]. A broken system (pre-fix: the typed class ignoring nice= on a
    # data-derived domain) renders the raw column extent instead — its tick
    # set would never contain the widened "0"/"10" bounds.
    assert "0" in typed_ticks and "10" in typed_ticks, (
        f"data domain [0.5, 9.5] with nice=True must render the niced [0, 10] "
        f"bounds as ticks, got {typed_ticks}"
    )
    assert "0" not in nonice_ticks or "10" not in nonice_ticks, (
        f"the nice=False baseline must NOT already contain both widened bounds "
        f"(or this domain doesn't actually discriminate), got {nonice_ticks}"
    )


def test_nice_renders_niced_ticks_for_typed_time_scale_with_no_domain():
    """#153's disclosed behavior change for `TimeScale`, pinned at render
    time — the sibling of the `LinearScale` pin above, closing the
    render-tick coverage gap the spec review flagged (Time/Log/Symlog were
    previously pinned only at the Rust wire-flag level, not rendered SVG).

    Data domain 2024-01-02T03:17:00Z .. 2024-01-04T09:42:00Z (neither
    endpoint on an hour boundary): calendar-nice widens both endpoints
    outward to the nearest hour, so the typed class's rendered ticks must
    match its raw-dict equivalent AND differ from the un-niced baseline
    (measured: niced ticks start at "03:00"; the un-niced baseline starts
    later, at "06:00", since it never widens past the data's actual start).
    """
    import datetime

    from ferrum._core import TimeScale
    from tests._svg_extents import axis_tick_labels

    lo = datetime.datetime(2024, 1, 2, 3, 17, 0, tzinfo=datetime.timezone.utc)
    hi = datetime.datetime(2024, 1, 4, 9, 42, 0, tzinfo=datetime.timezone.utc)
    mid = lo + (hi - lo) / 2
    df = pl.DataFrame({"x": [lo, mid, hi], "y": [1.0, 2.0, 3.0]})

    svg_typed = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=TimeScale(nice=True)), y="y").to_svg()
    )
    svg_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "nice": True}), y="y")
        .to_svg()
    )
    svg_nonice = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=TimeScale(nice=False)), y="y").to_svg()
    )
    typed_ticks = axis_tick_labels(svg_typed, axis="x")
    dict_ticks = axis_tick_labels(svg_dict, axis="x")
    nonice_ticks = axis_tick_labels(svg_nonice, axis="x")
    assert typed_ticks == dict_ticks, (
        f"typed TimeScale(nice=True) ticks {typed_ticks} must match the raw-dict "
        f"equivalent's ticks {dict_ticks}"
    )
    assert typed_ticks != nonice_ticks, (
        f"nice=True ticks {typed_ticks} must differ from the nice=False baseline {nonice_ticks} "
        f"— otherwise nice= isn't actually reaching the render path"
    )
    assert typed_ticks[0] == "03:00", (
        f"calendar-nice must widen the domain's start back to the 03:00 hour boundary, "
        f"got first tick {typed_ticks[0]!r} (full: {typed_ticks})"
    )
    assert nonice_ticks[0] != "03:00", (
        f"the nice=False baseline must NOT already start at 03:00 (or this domain doesn't "
        f"actually discriminate), got {nonice_ticks}"
    )


def test_nice_renders_niced_ticks_for_typed_log_scale_with_no_domain():
    """#153's disclosed behavior change for `LogScale`, pinned at render
    time — see `test_nice_renders_niced_ticks_for_typed_time_scale_with_no_domain`
    above for why this coverage exists (spec-review finding, render-tick
    parity across all four typed scale classes).

    Data domain [3, 700] (neither endpoint an exact power of 10): nice()
    rounds outward to the enclosing decades (`[1, 1000]`, matching the Rust
    `log_nice_rounds_to_decades` unit test), so the typed class's rendered
    ticks must match its raw-dict equivalent AND differ from the un-niced
    baseline (measured: niced ticks start exactly at "1"; the un-niced
    baseline starts inside the data's own range, at "3.9811").
    """
    from ferrum._core import LogScale
    from tests._svg_extents import axis_tick_labels

    df = pl.DataFrame({"x": [3.0, 350.0, 700.0], "y": [1.0, 2.0, 3.0]})

    svg_typed = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=LogScale(nice=True)), y="y").to_svg()
    )
    svg_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "log", "nice": True}), y="y")
        .to_svg()
    )
    svg_nonice = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=LogScale(nice=False)), y="y").to_svg()
    )
    typed_ticks = axis_tick_labels(svg_typed, axis="x")
    dict_ticks = axis_tick_labels(svg_dict, axis="x")
    nonice_ticks = axis_tick_labels(svg_nonice, axis="x")
    assert typed_ticks == dict_ticks, (
        f"typed LogScale(nice=True) ticks {typed_ticks} must match the raw-dict "
        f"equivalent's ticks {dict_ticks}"
    )
    assert typed_ticks != nonice_ticks, (
        f"nice=True ticks {typed_ticks} must differ from the nice=False baseline {nonice_ticks} "
        f"— otherwise nice= isn't actually reaching the render path"
    )
    assert typed_ticks[0] == "1", (
        f"nice=True must round the domain down to the enclosing decade (1), "
        f"got first tick {typed_ticks[0]!r} (full: {typed_ticks})"
    )
    assert nonice_ticks[0] != "1", (
        f"the nice=False baseline must NOT already start at the decade boundary 1 "
        f"(or this domain doesn't actually discriminate), got {nonice_ticks}"
    )


def test_nice_renders_niced_ticks_for_typed_symlog_scale_with_no_domain():
    """#153's disclosed behavior change for `SymlogScale`, pinned at render
    time — see `test_nice_renders_niced_ticks_for_typed_time_scale_with_no_domain`
    above for why this coverage exists (spec-review finding, render-tick
    parity across all four typed scale classes).

    Same repro domain as the `LinearScale` pin ([0.5, 9.5] -> niced [0, 10]):
    `SymlogScale.nice()` uses the identical `nice_step`-count-10 rounding
    `LinearScaleData::nice` does, so the typed class's rendered ticks must
    match its raw-dict equivalent AND differ from the un-niced baseline.
    """
    from ferrum._core import SymlogScale
    from tests._svg_extents import axis_tick_labels

    df = pl.DataFrame({"x": [0.5, 5.0, 9.5], "y": [1.0, 2.0, 3.0]})

    svg_typed = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=SymlogScale(nice=True)), y="y").to_svg()
    )
    svg_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "symlog", "nice": True}), y="y")
        .to_svg()
    )
    svg_nonice = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=SymlogScale(nice=False)), y="y").to_svg()
    )
    typed_ticks = axis_tick_labels(svg_typed, axis="x")
    dict_ticks = axis_tick_labels(svg_dict, axis="x")
    nonice_ticks = axis_tick_labels(svg_nonice, axis="x")
    assert typed_ticks == dict_ticks, (
        f"typed SymlogScale(nice=True) ticks {typed_ticks} must match the raw-dict "
        f"equivalent's ticks {dict_ticks}"
    )
    assert typed_ticks != nonice_ticks, (
        f"nice=True ticks {typed_ticks} must differ from the nice=False baseline {nonice_ticks} "
        f"— otherwise nice= isn't actually reaching the render path"
    )
    assert "0" in typed_ticks and "10" in typed_ticks, (
        f"data domain [0.5, 9.5] with nice=True must render the niced [0, 10] "
        f"bounds as ticks, got {typed_ticks}"
    )
    assert "0" not in nonice_ticks or "10" not in nonice_ticks, (
        f"the nice=False baseline must NOT already contain both widened bounds "
        f"(or this domain doesn't actually discriminate), got {nonice_ticks}"
    )


# ---------------------------------------------------------------------------
# F16: reverse= on positional scales serializes
# ---------------------------------------------------------------------------


def test_reverse_serialized_point_scale():
    """PointScale(reverse=True) serializes reverse into the spec dict."""
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import PointScale

    s = PointScale(domain=["a", "b", "c"], range=[0, 300], reverse=True)
    d = _scale_to_dict(s)
    assert d.get("type") == "point"
    assert d.get("reverse") is True, "reverse was not serialized for PointScale"
    # Regression: issue #39 — PointScale range was silently dropped from the
    # wire dict (ScaleSpec::Point had no range field).
    assert d.get("range") == [0.0, 300.0], "range was not serialized for PointScale"


def test_reverse_serialized_sequential_scale():
    """SequentialScale(reverse=True) serializes reverse into the spec dict."""
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import SequentialScale

    s = SequentialScale(scheme="viridis", domain=[0.0, 1.0], reverse=True)
    d = _scale_to_dict(s)
    assert d.get("type") == "sequential"
    assert d.get("reverse") is True, "reverse was not serialized for SequentialScale"


def test_reverse_false_is_serialized():
    """reverse=False is included in the spec dict (it is the non-default for some types)."""
    from ferrum.encoding._scale import _scale_to_dict
    from ferrum._core import PointScale

    s = PointScale(domain=["a", "b"], range=[0, 200], reverse=False)
    d = _scale_to_dict(s)
    assert "reverse" in d
    assert d["reverse"] is False


def test_reverse_dict_scale_passes_through():
    """A dict scale with reverse=True passes through _scale_to_dict unchanged."""
    from ferrum.encoding._scale import _scale_to_dict

    d = _scale_to_dict({"type": "linear", "reverse": True})
    assert d.get("reverse") is True


# ---------------------------------------------------------------------------
# F17: Axis label remapping via Axis(label_map=...)
# ---------------------------------------------------------------------------


def test_axis_label_map_stored():
    """Axis(label_map=...) stores the mapping on the object."""
    ax = fm.Axis(label_map={"a": "Alpha", "b": "Beta"})
    assert ax.label_map == {"a": "Alpha", "b": "Beta"}


def test_axis_label_map_not_emitted_to_dict():
    """label_map is a Python-only field and must not appear in to_dict()."""
    ax = fm.Axis(label_map={"a": "Alpha"})
    d = ax.to_dict()
    assert "label_map" not in d


def test_axis_label_map_none_to_dict_unchanged():
    """Axis with no label_map produces the same to_dict() as before."""
    ax = fm.Axis(title="Category")
    d = ax.to_dict()
    assert d == {"title": "Category"}
    assert "label_map" not in d


def test_axis_labels_remapping_renders():
    """Axis(label_map=...) remaps tick labels in the rendered SVG."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(
            x=fm.X("cat:N", axis=fm.Axis(label_map={"a": "Alpha", "b": "Beta", "c": "Gamma"})),
            y="val:Q",
        )
    )
    svg = chart.to_svg()
    assert "<svg" in svg
    assert "Alpha" in svg
    assert "Beta" in svg
    assert "Gamma" in svg


def test_axis_labels_remapping_partial():
    """label_map only remaps specified keys; unlisted values pass through unchanged."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(
            x=fm.X("cat:N", axis=fm.Axis(label_map={"a": "Alpha"})),
            y="val:Q",
        )
    )
    svg = chart.to_svg()
    assert "<svg" in svg
    assert "Alpha" in svg
    # b and c should still appear as-is
    assert "b" in svg
    assert "c" in svg


def test_axis_labels_bool_still_works():
    """Axis(labels=False) still suppresses tick labels (existing behaviour)."""
    df = pl.DataFrame({"cat": ["a", "b"], "val": [1.0, 2.0]})
    svg_with = fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q").to_svg()
    svg_without = (
        fm.Chart(df)
        .mark_bar()
        .encode(x=fm.X("cat:N", axis=fm.Axis(labels=False)), y="val:Q")
        .to_svg()
    )
    assert "<svg" in svg_with
    assert "<svg" in svg_without


# ---------------------------------------------------------------------------
# M5: annotate_abline
# ---------------------------------------------------------------------------


def test_annotate_abline_exported():
    """annotate_abline is exported from the top-level ferrum namespace."""
    assert hasattr(fm, "annotate_abline")


def test_annotate_abline_returns_chart():
    """annotate_abline returns a Chart."""
    line = fm.annotate_abline(slope=2.0, intercept=0.0)
    assert isinstance(line, fm.Chart)


def test_annotate_abline_renders():
    """annotate_abline renders to a valid SVG string."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    scatter = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    line = fm.annotate_abline(slope=2.0, intercept=0.0, stroke="red")
    combined = scatter + line
    svg = combined.to_svg()
    assert "<svg" in svg


def test_annotate_abline_stroke_in_spec():
    """The stroke colour passed to annotate_abline is present in the ChartSpec JSON."""
    line = fm.annotate_abline(slope=2.0, intercept=0.0, stroke="#ff0000")
    spec = line.to_spec()
    assert "#ff0000" in spec.to_json()


def test_annotate_abline_standalone_renders():
    """annotate_abline can render as a standalone chart (not only layered)."""
    line = fm.annotate_abline(slope=1.0, intercept=0.0, stroke="#333333")
    svg = line.to_svg()
    assert "<svg" in svg


def test_annotate_abline_stroke_dash():
    """annotate_abline with stroke_dash does not crash."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    scatter = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    line = fm.annotate_abline(slope=1.0, intercept=0.0, stroke_dash=[4, 4])
    svg = (scatter + line).to_svg()
    assert "<svg" in svg


def test_annotate_abline_identity_line():
    """annotate_abline(slope=1, intercept=0) renders the identity line without error."""
    df = pl.DataFrame({"x": [0.0, 0.5, 1.0], "y": [0.1, 0.4, 0.9]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    line = fm.annotate_abline(slope=1.0, intercept=0.0, stroke="gray")
    svg = (chart + line).to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# F1: .labs() fluent method
# ---------------------------------------------------------------------------


def test_labs_title():
    """labs(title=) sets the chart title visible in the SVG."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").labs(title="My Title")
    svg = chart.to_svg()
    assert "My Title" in svg


def test_labs_axis_labels():
    """labs(x=, y=) sets axis title labels visible in the SVG."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").labs(x="Custom X", y="Custom Y")
    svg = chart.to_svg()
    assert "Custom X" in svg
    assert "Custom Y" in svg


def test_labs_subtitle():
    """labs(subtitle=) sets the chart subtitle visible in the SVG."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").labs(subtitle="A subtitle")
    svg = chart.to_svg()
    assert "A subtitle" in svg


def test_labs_unknown_key_raises():
    """labs() with an unrecognised key raises ValueError."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    with pytest.raises(ValueError, match="unknown label keys"):
        fm.Chart(df).mark_point().encode(x="x", y="y").labs(z="Bad Key")


def test_labs_preserves_existing_channel_type():
    """labs(x=) on a typed channel keeps the type encoding intact."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", type="Q"), y="y").labs(x="X Axis Label")
    svg = chart.to_svg()
    assert "X Axis Label" in svg


def test_labs_x_on_channel_without_existing_encoding():
    """labs(x=) when x is not yet encoded creates a title-only channel that renders."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    # x is encoded via shorthand; labs sets an explicit title override
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").labs(x="My X")
    svg = chart.to_svg()
    assert "My X" in svg


def test_labs_returns_new_chart():
    """labs() is immutable — returns a new Chart, not mutating self."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    original = fm.Chart(df).mark_point().encode(x="x", y="y")
    labeled = original.labs(title="Title")
    assert original._title is None or original._title != labeled._title


# ---------------------------------------------------------------------------
# F2: .xlim() / .ylim() fluent methods
# ---------------------------------------------------------------------------


def test_xlim_ylim_renders():
    """xlim() and ylim() produce a chart that renders without error."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").xlim(0, 10).ylim(0, 20)
    svg = chart.to_svg()
    assert "<svg" in svg


def test_xlim_is_immutable():
    """xlim() returns a new Chart and does not mutate the original."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    original = fm.Chart(df).mark_point().encode(x="x", y="y")
    limited = original.xlim(0, 5)
    assert original._coord is None
    assert limited._coord is not None


def test_ylim_is_immutable():
    """ylim() returns a new Chart and does not mutate the original."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    original = fm.Chart(df).mark_point().encode(x="x", y="y")
    limited = original.ylim(0, 10)
    assert original._coord is None
    assert limited._coord is not None


def test_xlim_wires_coord_cartesian():
    """xlim() stores a CoordCartesian with xlim set on the chart."""
    from ferrum.coord import CoordCartesian

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").xlim(2.0, 8.0)
    assert isinstance(chart._coord, CoordCartesian)
    assert chart._coord.xlim == (2.0, 8.0)


def test_ylim_wires_coord_cartesian():
    """ylim() stores a CoordCartesian with ylim set on the chart."""
    from ferrum.coord import CoordCartesian

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").ylim(1.0, 9.0)
    assert isinstance(chart._coord, CoordCartesian)
    assert chart._coord.ylim == (1.0, 9.0)


def test_xlim_ylim_combined_wires_both():
    """xlim().ylim() chains correctly — both limits are set in the final coord."""
    from ferrum.coord import CoordCartesian

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").xlim(0, 5).ylim(0, 10)
    assert isinstance(chart._coord, CoordCartesian)
    assert chart._coord.xlim == (0, 5)
    assert chart._coord.ylim == (0, 10)


# ---------------------------------------------------------------------------
# F15: .to_dict() on Chart
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict():
    """to_dict() returns a plain Python dict."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    d = fm.Chart(df).mark_point().encode(x="x", y="y").to_dict()
    assert isinstance(d, dict)


def test_to_dict_contains_mark():
    """to_dict() result includes the mark key."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    d = fm.Chart(df).mark_point().encode(x="x", y="y").to_dict()
    assert "mark" in d
    assert d["mark"] == "point"


def test_to_dict_consistent_with_to_json():
    """to_dict() and json.loads(to_json()) produce identical structures."""
    import json

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    assert chart.to_dict() == json.loads(chart.to_json())


# ---------------------------------------------------------------------------
# B7: config defaults used by _render_inputs (width/height)
# ---------------------------------------------------------------------------


def test_config_defaults_render():
    """A chart renders using config width/height when no explicit dimensions set."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg = chart.to_svg()
    assert "<svg" in svg


def test_config_override_used_in_render():
    """ferrum.config.set(width=..., height=...) is honored at render time."""
    import ferrum.config as fc

    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")

    with fc.defaults(width=800, height=600):
        svg = chart.to_svg()

    # SVG should declare the overridden dimensions
    assert 'width="800"' in svg
    assert 'height="600"' in svg


def test_config_default_dimensions_match_config_module():
    """The built-in render fallback matches ferrum.config defaults (640×480)."""
    import ferrum.config as fc

    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg = chart.to_svg()

    expected_width = str(int(fc.get("width")))
    expected_height = str(int(fc.get("height")))
    assert f'width="{expected_width}"' in svg
    assert f'height="{expected_height}"' in svg


# ---------------------------------------------------------------------------
# WASM conditional encoding on extended channels
# ---------------------------------------------------------------------------


def test_conditional_stroke_opacity_round_trip():
    """StrokeOpacity conditional should serialize with kind=stroke_opacity, not opacity."""
    import json

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    sel = fm.selection_point(name="sel")
    cond = sel.when(fm.StrokeOpacity("x")).otherwise(fm.value(0.2))
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").add_selection(sel).conditional(cond)
    d = chart.to_dict()
    conds = d.get("conditionals", [])
    if isinstance(conds, str):
        conds = json.loads(conds)
    assert len(conds) == 1
    assert conds[0]["channel"] == "stroke_opacity"
    assert conds[0]["if_not"]["kind"] == "stroke_opacity"


def test_conditional_fill_opacity_round_trip():
    """FillOpacity conditional should serialize with kind=fill_opacity."""
    import json

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    sel = fm.selection_point(name="sel")
    cond = sel.when(fm.FillOpacity("x")).otherwise(fm.value(0.3))
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").add_selection(sel).conditional(cond)
    d = chart.to_dict()
    conds = d.get("conditionals", [])
    if isinstance(conds, str):
        conds = json.loads(conds)
    assert len(conds) == 1
    assert conds[0]["channel"] == "fill_opacity"
    assert conds[0]["if_not"]["kind"] == "fill_opacity"


def test_conditional_angle_round_trip():
    """Angle conditional should serialize with kind=angle."""
    import json

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    sel = fm.selection_point(name="sel")
    cond = sel.when(fm.Angle("x")).otherwise(fm.value(45.0))
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").add_selection(sel).conditional(cond)
    d = chart.to_dict()
    conds = d.get("conditionals", [])
    if isinstance(conds, str):
        conds = json.loads(conds)
    assert len(conds) == 1
    assert conds[0]["channel"] == "angle"
    assert conds[0]["if_not"]["kind"] == "angle"


# ---------------------------------------------------------------------------
# Regression tests for Rust-backed fixes lacking Python e2e coverage
# ---------------------------------------------------------------------------


def test_css_named_color_in_theme():
    """Regression: Theme(mark_color="steelblue") crashed with 'invalid color
    string' because parse_hex only accepted #rrggbb. Now parse_color handles
    all 148 CSS named colors."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .theme(fm.Theme(mark_color="steelblue"))
        .to_svg()
    )
    assert "4682b4" in svg


def test_css_named_color_cornflowerblue():
    """Regression: verify a second named color to guard against partial table."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .theme(fm.Theme(mark_color="cornflowerblue"))
        .to_svg()
    )
    assert "6495ed" in svg


def test_css_named_color_invalid_gives_helpful_error():
    """Regression: unknown color names should mention 'CSS color name'."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(ValueError, match="CSS color name"):
        fm.Chart(df).mark_point().encode(x="x", y="y").theme(
            fm.Theme(mark_color="notacolor")
        ).to_svg()


def test_mark_tick_ordinal_y_only():
    """Regression: mark_tick with only ordinal y produced empty SVG because
    tick.rs only handled quantitative y (y-rug). Now ordinal-y-only emits
    horizontal crossbars."""
    df = pl.DataFrame({"cat": ["a", "b", "c"]})
    svg = fm.Chart(df).mark_tick().encode(y="cat:N").to_svg()
    assert svg.count("<line") >= 3


def test_mark_tick_ordinal_x_only():
    """Regression: same fix also added ordinal-x-only vertical crossbars."""
    df = pl.DataFrame({"cat": ["a", "b", "c"]})
    svg = fm.Chart(df).mark_tick().encode(x="cat:N").to_svg()
    assert svg.count("<line") >= 3


def test_point_shape_vline():
    """Regression: mark_point(shape='|') fell back to circle because
    ShapeKind had no VLine variant."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    svg = fm.Chart(df).mark_point(shape="|").encode(x="x", y="y").to_svg()
    assert "<line" in svg
    assert "<circle" not in svg


def test_point_shape_hline():
    """Regression: mark_point(shape='-') fell back to circle."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    svg = fm.Chart(df).mark_point(shape="-").encode(x="x", y="y").to_svg()
    assert "<line" in svg
    assert "<circle" not in svg


def test_point_shape_vline_alias():
    """Regression: 'vline' string alias for '|' shape."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    svg = fm.Chart(df).mark_point(shape="vline").encode(x="x", y="y").to_svg()
    assert "<line" in svg


def test_mark_smooth_method_linear():
    """Regression: mark_smooth(method='linear') crashed because smooth.rs only
    accepted 'lm' and 'loess'. Now 'linear' is an alias for 'lm'."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    svg = fm.Chart(df).mark_smooth(method="linear").encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    assert "<path" in svg or "d=" in svg


def test_mark_smooth_method_quadratic():
    """Regression: 'quadratic' is a new polynomial degree-2 method."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    svg = fm.Chart(df).mark_smooth(method="quadratic").encode(x="x", y="y").to_svg()
    assert "<svg" in svg


def test_mark_smooth_method_lm_still_works():
    """Regression: original 'lm' method must still work after alias additions."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    svg = fm.Chart(df).mark_smooth(method="lm").encode(x="x", y="y").to_svg()
    assert "<svg" in svg


def test_non_int64_nominal_pyarrow():
    """Regression: pyarrow Int32/UInt8 columns encoded as :N crashed with
    'unsupported dtype: Int32 (cannot enumerate distinct values)' because
    distinct_values_in_order only handled Int64."""
    import pyarrow as pa

    tbl = pa.table({"cat": pa.array([1, 2, 1, 2], type=pa.int32()), "y": [10.0, 20.0, 30.0, 40.0]})
    svg = fm.Chart(tbl).mark_bar().encode(x="cat:N", y="y:Q").to_svg()
    assert "<rect" in svg


def test_non_int64_nominal_uint8():
    """Regression: UInt8 nominal should also work."""
    import pyarrow as pa

    tbl = pa.table({"g": pa.array([0, 1, 0, 1], type=pa.uint8()), "v": [1.0, 2.0, 3.0, 4.0]})
    svg = fm.Chart(tbl).mark_bar().encode(x="g:N", y="v:Q").to_svg()
    assert "<rect" in svg


def test_aggregate_variance():
    """Regression: encode(y='variance(val):Q') crashed because AggFn only had
    6 variants. Now variance/stdev/q1/q3/distinct are supported."""
    df = pl.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 3.0, 10.0, 20.0]})
    svg = fm.Chart(df).mark_bar().encode(x="cat:N", y="variance(val):Q").to_svg()
    assert "<rect" in svg


def test_aggregate_stdev():
    """Regression: stdev aggregate function."""
    df = pl.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 3.0, 10.0, 20.0]})
    svg = fm.Chart(df).mark_bar().encode(x="cat:N", y="stdev(val):Q").to_svg()
    assert "<rect" in svg


def test_aggregate_q1_q3():
    """Regression: q1/q3 aggregate functions."""
    df = pl.DataFrame(
        {
            "cat": ["a"] * 5 + ["b"] * 5,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    svg_q1 = fm.Chart(df).mark_bar().encode(x="cat:N", y="q1(val):Q").to_svg()
    svg_q3 = fm.Chart(df).mark_bar().encode(x="cat:N", y="q3(val):Q").to_svg()
    assert "<rect" in svg_q1
    assert "<rect" in svg_q3


def test_aggregate_distinct():
    """Regression: distinct aggregate function (count unique)."""
    df = pl.DataFrame({"cat": ["a", "a", "b", "b", "b"], "val": [1.0, 1.0, 2.0, 3.0, 3.0]})
    svg = fm.Chart(df).mark_bar().encode(x="cat:N", y="distinct(val):Q").to_svg()
    assert "<rect" in svg


def test_kde_kernel_epanechnikov():
    """Regression: mark_density(kernel='epanechnikov') raised ValueError
    because desugar_density rejected non-gaussian AND never forwarded kernel
    to Kde(). Both are now fixed."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.5]})
    svg = fm.Chart(df).mark_density(kernel="epanechnikov").encode(x="val:Q").to_svg()
    assert "<svg" in svg
    assert "<path" in svg or "d=" in svg


def test_kde_kernel_tophat():
    """Regression: tophat (uniform) kernel."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    svg = fm.Chart(df).mark_density(kernel="tophat").encode(x="val:Q").to_svg()
    assert "<svg" in svg


def test_kde_kernel_cosine():
    """Regression: cosine kernel."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    svg = fm.Chart(df).mark_density(kernel="cosine").encode(x="val:Q").to_svg()
    assert "<svg" in svg


def test_kde_kernel_gaussian_still_default():
    """Regression: default kernel (gaussian) must still work."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    svg = fm.Chart(df).mark_density().encode(x="val:Q").to_svg()
    assert "<svg" in svg


def test_kde_kernel_invalid_rejected():
    """Regression: unknown kernel names should raise ValueError."""
    df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match=r"mark_density: kernel must be one of"):
        fm.Chart(df).mark_density(kernel="banana").encode(x="val:Q").to_svg()
