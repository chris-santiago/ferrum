"""Rendered-output pins for D8's Rust-half format_type threading (batch B Task 4).

``test_format_preset_resolution.py`` (Task 2) pins that a preset name never
reaches Rust unresolved on all five Python emission surfaces, and explicitly
scopes its time-preset coverage to the *wire* — the module docstring there
says "Rendered date formatting at the axis/legend level requires format_type
consumption that lands in batch B Task 4". This module is that consumption's
render-level pin:

- Chart-level time presets (``configure_axis(label_format="date_iso")``)
  render real dates instead of misparsing the resolved strftime pattern as a
  d3 numeric spec (the ``%`` character was read as the d3 percent TYPE char,
  producing output like ``"300.000000%"`` — ``render/mod.rs``'s former
  hardcoded ``format_type: None``).
- Cascade discipline holds: an explicit per-channel time format
  (``fm.Axis(label_format=...)``) ALWAYS beats a chart-level one, even
  though the per-channel temporal path applies its format eagerly and
  threads ``None`` back (spec-review cycle 2 finding — a chart-level
  ``date_iso`` preset was winning the cascade by re-deriving raw temporal
  values and overwriting the already-formatted per-channel labels;
  ``AxisStyleOverrides.label_format_claimed`` now marks the axis as claimed
  independently of whether a string survived to be threaded).
- ``"ordinal"`` renders real 1st/2nd/3rd suffixes (was previously a no-op:
  the ``"__ordinal__"`` sentinel had no Rust consumer at all).
- ``label_format`` + explicit ``values=`` compose in both directions
  (NF-B2), on both the per-channel and chart-level surfaces.
- ``LegendStyleSpec.format_type`` (colorbar/size-legend) is read — the
  legend half of the same threading.
- The T2-discovered residual: a typo'd preset name (``"curency"``) still
  reproduces NF-B1's control-character harm on the four raw-accepting
  surfaces, because §4.5's unknown-name-passes-raw contract is correct at
  Python and the harm lived in Rust's lenient d3-spec parser silently
  discarding trailing garbage after a spurious type-char match. Each surface
  now gets a typed ``ValueError`` refusal instead.

Every temporal assertion below uses a genuinely temporal ``:T`` column
(``_temporal_df`` / ``_cascade_df``), never a plain integer x/y frame
(spec-review cycle 2 finding: on an integer-only frame, small values like
``1..11`` alias to 1970-epoch-ms under a time format via
``apply_tick_format``'s string-reparse fallback, so an ISO-shaped assertion
can pass from that fallback alone — it would still pass with the real
``scale.temporal_tick_values`` re-derivation branch deleted entirely, which
is exactly why the cascade-inversion bug above went unnoticed by the first
cycle's tests).

Cycle-3 additions (quality-review fix round):

- A malformed ``%``-bearing spec on a genuinely temporal channel (e.g. the
  typo'd preset class ``"curency%"``, or an ordinary raw d3 percent spec
  like ``".1%"`` that happens to auto-detect as a time candidate) is a typed
  ``ValueError``, never a Rust panic crossing the PyO3 boundary —
  ``validate_chart_format_specs`` previously exempted every ``%``-bearing
  spec from validation on the false premise that ``chrono`` handles a bad
  strftime pattern "leniently"; it panics instead.
- A chart-level time format survives ``tick_min_step`` thinning instead of
  silently reverting to the default granularity.
- ``fm.Legend(format_type=...)`` alone (a non-``"time"`` value, no
  ``format=``) renders byte-identically to the untouched default.
"""

from __future__ import annotations

import datetime as dt
import re
from datetime import date

import polars as pl
import pytest

import ferrum as fm
from ferrum.configure import AxisConfig
from tests._snapshots import control_chars as _control_chars
from tests._snapshots import legend_texts as _extract_text_labels


def _xy_df() -> pl.DataFrame:
    return pl.DataFrame({"x": list(range(1, 12)), "y": [float(i) for i in range(1, 12)]})


def _temporal_df() -> pl.DataFrame:
    """A genuinely temporal x column (six real dates, 2020-01 .. 2020-06)
    paired with a plain numeric y — see the module docstring's discriminating
    note on why this fixture (not `_xy_df`) is required for every temporal
    assertion in this module. At this span the DEFAULT (unformatted) tick
    granularity is weekly/daily (`YYYY-MM-DD`-shaped), which happens to
    coincide with `date_iso`'s own output — fine for `month_year` (whose
    `"%b %Y"` output is never day-shaped either way) but NOT
    discriminating for `date_iso` itself; see `_wide_temporal_df` for that."""
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 6, 1), "1mo", eager=True),
            "val": [float(i) for i in range(6)],
        }
    )


def _wide_temporal_df() -> pl.DataFrame:
    """A wide (four-year) temporal x column whose DEFAULT tick granularity is
    month/year (`"Jan 2018"`-shaped, from `format::format_time`'s own
    spacing-keyed default) — clearly distinct from `date_iso`'s forced
    day-level `"YYYY-MM-DD"` output, so a `date_iso`-formatted render can
    only match this shape via the real `scale.temporal_tick_values`
    re-derivation path, never by coincidence with the default (spec-review
    cycle 2 finding: `_temporal_df`'s narrower span produces
    coincidentally-ISO-shaped default labels, under which `date_iso`
    assertions pass even with the whole re-derivation branch deleted)."""
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2018, 1, 1), date(2022, 1, 1), "3mo", eager=True),
            "val": [float(i) for i in range(17)],
        }
    )


def _time_df(n: int = 10, step_days: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": list(range(n)),
            "y": [float(i) for i in range(n)],
            "t": [dt.datetime(2020, 1, 1) + dt.timedelta(days=i * step_days) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Chart-level time presets render real dates (the batch's motivating repro).
# ---------------------------------------------------------------------------


def test_chart_level_date_iso_preset_renders_dates_not_percent():
    df = _wide_temporal_df()
    default_svg = fm.Chart(df).mark_point().encode(x="date:T", y="val:Q").to_svg()
    default_labels = _extract_text_labels(default_svg)
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="date:T", y="val:Q")
        .configure_axis(label_format="date_iso")
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    date_labels = {l for l in labels if re.match(r"^\d{4}-\d{2}-\d{2}$", l)}
    # Multiple DISTINCT real dates, at least one genuinely in 2020 — not a
    # single repeated "1970-01-01" epoch artifact (the y-axis, also hit by
    # `configure_axis` applying to both axes, legitimately produces that
    # artifact; the x-axis must not).
    assert len(date_labels) > 1, f"expected multiple distinct ISO date labels, got: {labels}"
    assert any(l.startswith("2020-") for l in date_labels), f"expected 2020 dates: {date_labels}"
    # The pre-fix bug: '%' misparsed as the d3 percent type char.
    assert not any("%" in l for l in labels), f"date labels must not carry '%': {labels}"
    # Must actually be doing something -- `_wide_temporal_df`'s default
    # granularity is "Jan 2018"-shaped, never coincidentally ISO-shaped.
    assert set(default_labels) != set(labels), "date_iso rendered identically to the default"


def test_chart_level_axis_x_date_iso_preset_renders_dates():
    """Same fix via the per-axis `axis_x=AxisConfig(...)` position, scoped
    to x only (sidesteps the y-axis both-axes-application artifact noted
    above, which is a separate, pre-existing, out-of-scope concern)."""
    df = _wide_temporal_df()
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="date:T", y="val:Q")
        .configure(axis_x=AxisConfig(label_format="date_iso"))
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    date_labels = {l for l in labels if re.match(r"^\d{4}-\d{2}-\d{2}$", l)}
    assert len(date_labels) > 1, f"expected multiple distinct ISO date labels, got: {labels}"
    assert any(l.startswith("2020-") for l in date_labels), f"expected 2020 dates: {date_labels}"


def test_chart_level_month_year_preset_renders_month_names():
    svg = (
        fm.Chart(_temporal_df())
        .mark_point()
        .encode(x="date:T", y="val:Q")
        .configure(axis_x=AxisConfig(label_format="month_year"))
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    month_labels = {l for l in labels if re.match(r"^[A-Z][a-z]{2} \d{4}$", l)}
    assert len(month_labels) > 1, f"expected multiple distinct month/year labels, got: {labels}"


# ---------------------------------------------------------------------------
# Cascade discipline (spec-review cycle 2): per-channel format ALWAYS wins
# over chart-level, even for a temporal per-channel format that applies
# eagerly and threads `label_format = None` back (nothing left to defer —
# `AxisStyleOverrides.label_format_claimed` is what actually protects it).
# ---------------------------------------------------------------------------


def _cascade_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 4, 1), "1mo", eager=True),
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_per_channel_time_format_alone_renders_its_own_pattern():
    svg = (
        fm.Chart(_cascade_df())
        .mark_point()
        .encode(x=fm.X("date:T", axis=fm.Axis(label_format="%m/%d")), y="val:Q")
        .to_svg()
    )
    labels = set(_extract_text_labels(svg))
    mmdd = {l for l in labels if re.match(r"^\d{2}/\d{2}$", l)}
    assert mmdd, f"expected %m/%d labels, got: {labels}"
    assert not any(re.match(r"^\d{4}-\d{2}-\d{2}$", l) for l in labels), labels


def test_per_channel_time_format_beats_chart_level_time_preset():
    """The exact cascade-inversion repro (spec-review cycle 2, live-verified
    on the pre-fix build): `.encode(x=fm.X("date:T", axis=fm.Axis(
    label_format="%m/%d")))` alone renders `%m/%d`-shaped labels; adding
    `.configure_axis(label_format="date_iso")` must NOT flip them to ISO
    dates — per-channel > chart-level, unconditionally."""
    chart = (
        fm.Chart(_cascade_df())
        .mark_point()
        .encode(x=fm.X("date:T", axis=fm.Axis(label_format="%m/%d")), y="val:Q")
    )
    labels_alone = _extract_text_labels(chart.to_svg())
    labels_with_chart_level = _extract_text_labels(
        chart.configure_axis(label_format="date_iso").to_svg()
    )
    mmdd_alone = {l for l in labels_alone if re.match(r"^\d{2}/\d{2}$", l)}
    mmdd_both = {l for l in labels_with_chart_level if re.match(r"^\d{2}/\d{2}$", l)}
    assert mmdd_alone, f"expected %m/%d labels alone, got: {labels_alone}"
    assert mmdd_both == mmdd_alone, (
        "chart-level date_iso must not override the per-channel %m/%d "
        f"format on x: alone={mmdd_alone}, with chart-level={mmdd_both}"
    )
    # No real (2020-prefixed) ISO date may appear anywhere -- that would
    # mean chart-level won the x-axis. (The y-axis's "1970-01-01" epoch
    # artifact from the separate, out-of-scope both-axes application is a
    # different, pre-existing concern and never 2020-prefixed.)
    assert not any(l.startswith("2020-") for l in labels_with_chart_level), labels_with_chart_level


def test_chart_level_time_preset_alone_wins_when_unclaimed():
    """Control: with no per-channel override at all, chart-level DOES fill
    and re-derive — the correct "chart-level alone" behavior the
    claimed-flag fix must not regress. Uses `_wide_temporal_df` (not
    `_cascade_df`) for the same reason as the date_iso tests above: at
    `_cascade_df`'s narrower span the default granularity already happens
    to be ISO-shaped, so this assertion would pass even with chart-level
    formatting completely inert."""
    df = _wide_temporal_df()
    default_svg = fm.Chart(df).mark_point().encode(x="date:T", y="val:Q").to_svg()
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="date:T", y="val:Q")
        .configure(axis_x=AxisConfig(label_format="date_iso"))
        .to_svg()
    )
    labels = {l for l in _extract_text_labels(svg) if re.match(r"^\d{4}-\d{2}-\d{2}$", l)}
    assert labels, f"expected ISO date labels, got: {_extract_text_labels(svg)}"
    assert any(l.startswith("2020-") for l in labels), labels
    assert _extract_text_labels(default_svg) != _extract_text_labels(svg), (
        "date_iso rendered identically to the default"
    )


# ---------------------------------------------------------------------------
# "ordinal" renders real suffixes, at chart level and per-channel.
# ---------------------------------------------------------------------------


def test_chart_level_ordinal_preset_renders_suffixes():
    svg = (
        fm.Chart(_xy_df())
        .mark_point()
        .encode(x="x", y="y")
        .configure_axis(label_format="ordinal")
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert any(re.match(r"^\d+(st|nd|rd|th)$", l) for l in labels), labels
    # Not a no-op: the default (unformatted) render has no ordinal suffixes.
    default_svg = fm.Chart(_xy_df()).mark_point().encode(x="x", y="y").to_svg()
    assert default_svg != svg


def test_per_channel_ordinal_preset_renders_suffixes():
    svg = (
        fm.Chart(_xy_df())
        .mark_point()
        .encode(x=fm.X("x", axis=fm.Axis(label_format="ordinal")), y="y")
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert any(re.match(r"^\d+(st|nd|rd|th)$", l) for l in labels), labels


def test_ordinal_teens_exception_renders_th_not_st():
    df = pl.DataFrame({"x": [11, 12, 13, 21], "y": [1.0, 2.0, 3.0, 4.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", axis=fm.Axis(label_format="ordinal", values=[11, 12, 13, 21])), y="y")
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert "11th" in labels
    assert "12th" in labels
    assert "13th" in labels
    assert "21st" in labels


# ---------------------------------------------------------------------------
# NF-B2: label_format + explicit values= compose, both directions.
# ---------------------------------------------------------------------------


def test_per_channel_percent_format_composes_with_explicit_values():
    svg = (
        fm.Chart(_xy_df())
        .mark_point()
        .encode(
            x=fm.X("x", axis=fm.Axis(label_format="percent", values=[0.0, 0.5, 1.0])),
            y="y",
        )
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert "0.0%" in labels
    assert "50.0%" in labels
    assert "100.0%" in labels


def test_chart_level_percent_format_composes_with_explicit_tick_values():
    svg = (
        fm.Chart(_xy_df())
        .mark_point()
        .encode(x="x", y="y")
        .configure(axis_x=AxisConfig(label_format="percent", tick_values=[0.0, 0.5, 1.0]))
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert "0.0%" in labels
    assert "50.0%" in labels
    assert "100.0%" in labels


# ---------------------------------------------------------------------------
# LegendStyleSpec.format_type — the legend half of the threading.
# ---------------------------------------------------------------------------


def test_color_legend_time_preset_renders_dates():
    svg = (
        fm.Chart(_time_df())
        .mark_point()
        .encode(x="x", y="y", color=fm.Color("t", legend=fm.Legend(format="date_iso")))
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert any(re.match(r"^\d{4}-\d{2}-\d{2}$", l) for l in labels), labels


def test_size_legend_time_preset_renders_dates():
    svg = (
        fm.Chart(_time_df())
        .mark_point()
        .encode(x="x", y="y", size=fm.Size("t", legend=fm.Legend(format="date_iso")))
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    assert any(re.match(r"^\d{4}-\d{2}-\d{2}$", l) for l in labels), labels


def test_color_legend_no_format_type_default_unaffected():
    """Byte-identity guard: a legend that never sets format/format_type
    renders identically to before this task (format_value_with_spec falls
    back to the pre-existing default colorbar-tick formatting).

    Quality-review S2 finding: this test's assertions previously only
    checked `'<svg ' in svg` and the absence of control characters, which
    every colorbar render satisfies regardless of what its tick labels
    actually say -- it carried this task's zero-golden-mover argument for
    the legend half without actually pinning any label text. Asserts the
    real, deterministic tick labels for the fixed [10, 20, 30] colorbar
    domain instead: 5 evenly-spaced values (10, 15, 20, 25, 30) formatted
    by the pre-existing range-aware `format_colorbar_tick`, unchanged from
    before this task."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "c": [10.0, 20.0, 30.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y", color="c").to_svg()
    assert not _control_chars(svg)
    labels = _extract_text_labels(svg)
    # The colorbar's title ("c") followed immediately by its 5 tick labels
    # are the last 6 <text> elements emitted.
    assert labels[-6:] == ["c", "10", "15", "20", "25", "30"], (
        f"colorbar title+tick labels changed from the pre-existing default: {labels}"
    )


# ---------------------------------------------------------------------------
# T2-discovered residual: malformed raw spec refusal, per raw-accepting
# surface — the "curency" typo repro.
# ---------------------------------------------------------------------------


def test_per_channel_axis_curency_typo_refused_not_corrupted():
    with pytest.raises(ValueError, match="curency"):
        fm.Chart(_xy_df()).mark_point().encode(
            x=fm.X("x", axis=fm.Axis(label_format="curency")), y="y"
        ).to_svg()


def test_encoding_format_curency_typo_refused_not_corrupted():
    with pytest.raises(ValueError, match="curency"):
        fm.Chart(_xy_df()).mark_point().encode(x=fm.X("x", format="curency"), y="y").to_svg()


def test_legend_format_curency_typo_refused_not_corrupted():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "c": [10.0, 20.0, 30.0]})
    with pytest.raises(ValueError, match="curency"):
        fm.Chart(df).mark_point().encode(
            x="x", y="y", color=fm.Color("c", legend=fm.Legend(format="curency"))
        ).to_svg()


def test_chart_level_label_format_raw_curency_typo_refused():
    """The chart-level raw-spelling surface (label_format_raw=, or a raw-dict
    caller) — AxisConfig's preset-name-only label_format= is already refused
    at Python construction time (test_format_preset_resolution.py), so this
    exercises the RAW spelling specifically.

    Quality-review cycle-5 fix (recurring at this site, cycle-3 finding):
    the wrong (misdiagnosed) "valid date/time pattern" message also
    satisfies a bare `match="curency"` check, since it echoes the spec
    string back — so a regression here would pass silently without the
    negative-control assertion below. "curency" has no '%' at all, so it
    can never legitimately be a date/time candidate; the refusal must stay
    the honest d3-grammar complaint.
    """
    with pytest.raises(ValueError, match="curency") as exc_info:
        fm.Chart(_xy_df()).mark_point().encode(x="x", y="y").configure_axis(
            label_format_raw="curency"
        ).to_svg()
    msg = str(exc_info.value)
    assert "unrecognized token" in msg, f"expected the d3-grammar message: {msg}"
    assert "valid date/time pattern" not in msg, (
        f"a %-free typo must never be misdiagnosed as a date/time pattern: {msg}"
    )


def test_chart_level_label_format_raw_real_strftime_gets_date_time_diagnosis():
    """The mirror-image positive case on the same surface: a genuine
    strftime pattern (has a real '%' specifier) IS diagnosed as a date/time
    pattern, pointing the user at the surfaces that actually accept one."""
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_xy_df()).mark_point().encode(x="x", y="y").configure_axis(
            label_format_raw="%b %d"
        ).to_svg()
    msg = str(exc_info.value)
    assert "valid date/time pattern" in msg, msg
    assert "fm.Axis(label_format=" in msg, msg
    assert "unrecognized token" not in msg, msg


def test_curency_refusal_is_typed_valueerror_not_silent_corruption():
    """Direct proof the refusal replaces the historical control-character
    corruption class, not just that SOME error fires."""
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_xy_df()).mark_point().encode(x=fm.X("x", format="curency"), y="y").to_svg()
    msg = str(exc_info.value)
    assert "curency" in msg
    assert not _control_chars(msg)


def test_valid_but_unusual_spec_still_renders():
    """The refusal must never false-positive on a genuinely valid, if
    exotic, d3 spec."""
    svg = fm.Chart(_xy_df()).mark_point().encode(x=fm.X("x", format="*>8.1%"), y="y").to_svg()
    assert "<svg " in svg
    assert not _control_chars(svg)


# ---------------------------------------------------------------------------
# Quality-review S4 (2026-09-03): malformed %-bearing specs on a genuinely
# TEMPORAL channel must be a typed ValueError, never a Rust panic crossing
# the PyO3 boundary. These are the exact repros quality review verified
# crash on the pre-fix build via chrono's DelayedFormat Display erroring
# inside format_time_spec's old .to_string() call.
# ---------------------------------------------------------------------------


def _monthly_temporal_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 4, 1), "1mo", eager=True),
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_curency_percent_typo_on_temporal_axis_raises_valueerror_not_panic():
    with pytest.raises(ValueError, match="curency%"):
        fm.Chart(_monthly_temporal_df()).mark_point().encode(
            x=fm.X("date:T", axis=fm.Axis(label_format="curency%")), y="val:Q"
        ).to_svg()


def test_unknown_strftime_specifier_on_temporal_axis_raises_valueerror():
    with pytest.raises(ValueError):
        fm.Chart(_monthly_temporal_df()).mark_point().encode(
            x=fm.X("date:T", axis=fm.Axis(label_format="%J")), y="val:Q"
        ).to_svg()


def test_ordinary_percent_encoding_format_on_temporal_axis_raises_valueerror():
    """`.1%` is a perfectly ordinary raw d3 percent spec — valid on a
    numeric scale, but not a valid strftime pattern, and `date:T` here IS
    declared temporal."""
    with pytest.raises(ValueError):
        fm.Chart(_monthly_temporal_df()).mark_point().encode(
            x=fm.X("date:T", format=".1%"), y="val:Q"
        ).to_svg()


def test_bare_percent_encoding_format_on_temporal_axis_raises_valueerror():
    with pytest.raises(ValueError):
        fm.Chart(_monthly_temporal_df()).mark_point().encode(
            x=fm.X("date:T", format="%"), y="val:Q"
        ).to_svg()


def test_ordinary_percent_encoding_format_on_non_temporal_axis_still_renders():
    """The exact mirror-image control: `.1%` on a non-temporal x must still
    render fine -- the fix must not become an over-eager refusal."""
    svg = (
        fm.Chart(_monthly_temporal_df())
        .mark_point()
        .encode(x=fm.X("val", format=".1%"), y="date:T")
        .to_svg()
    )
    assert "<svg " in svg
    assert not _control_chars(svg)


# ---------------------------------------------------------------------------
# Quality-review S3 (2026-09-03): a chart-level time format must survive
# tick_min_step thinning instead of silently reverting to the default
# granularity.
# ---------------------------------------------------------------------------


def test_chart_level_time_format_survives_tick_min_step_thinning():
    df = pl.DataFrame(
        {
            "date": pl.date_range(date(2018, 1, 1), date(2022, 1, 1), "3mo", eager=True),
            "val": [float(i) for i in range(17)],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="date:T", y="val:Q")

    unthinned_svg = chart.configure(axis_x=AxisConfig(label_format="date_iso")).to_svg()
    unthinned_dates = {
        l for l in _extract_text_labels(unthinned_svg) if re.match(r"^\d{4}-\d{2}-\d{2}$", l)
    }

    svg = chart.configure(
        axis_x=AxisConfig(label_format="date_iso", tick_min_step=1.5 * 365 * 86_400_000)
    ).to_svg()
    labels = _extract_text_labels(svg)
    date_labels = {l for l in labels if re.match(r"^\d{4}-\d{2}-\d{2}$", l)}

    assert date_labels, f"expected ISO date labels to survive thinning, got: {labels}"
    assert any(l.startswith("20") for l in date_labels), date_labels
    # The interaction this test is named for: thinning must actually have
    # happened, not just "the format survived whatever ticks were already
    # there" (quality-review S1, cycle 4 — this test previously never
    # asserted the tick count actually dropped, so it would keep passing
    # even if tick_min_step silently became inert).
    assert len(date_labels) < len(unthinned_dates), (
        f"tick_min_step did not thin the tick set: unthinned={sorted(unthinned_dates)}, "
        f"thinned={sorted(date_labels)}"
    )


def test_chart_level_time_format_without_tick_min_step_unaffected():
    """Control: the fix must not change output when tick_min_step is unset."""
    df = pl.DataFrame(
        {
            "date": pl.date_range(date(2018, 1, 1), date(2022, 1, 1), "3mo", eager=True),
            "val": [float(i) for i in range(17)],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="date:T", y="val:Q")
        .configure(axis_x=AxisConfig(label_format="date_iso"))
        .to_svg()
    )
    labels = _extract_text_labels(svg)
    date_labels = {l for l in labels if re.match(r"^\d{4}-\d{2}-\d{2}$", l)}
    assert len(date_labels) > 1, labels
