"""Format-preset resolution across all five emission surfaces (NF-B1, D8).

Before this fix, a preset name (e.g. ``"percent"``) reached the Rust
d3-format parser unresolved on four of the five surfaces — per-channel
``fm.Axis(label_format=)`` (``axis.py``), per-channel ``fm.Legend(format=)``
(``legend.py``), encoding ``format=`` (``encoding/base.py``), and the
raw-dict axis/legend normalize paths (``_normalize_axis`` /
``_normalize_legend``) — rendering literal control characters (``\\x1c``,
``\\x1a``, ``\\x18``) into user SVGs instead of a formatted number. Only the
chart-level ``AxisConfig`` already resolved. ``resolve_format_or_raw`` /
``resolve_format_field`` (``ferrum/format_presets.py``) are now the single
resolution point every surface routes through.

Numeric presets already work end to end once resolved (Rust's tick-format
consumer needs only the resolved d3-format string, not the newly-threaded
``format_type``), so this module pins *rendered* SVG output for them. Time
presets additionally require chart-level ``format_type`` consumption that
lands in batch B Task 4 (the Rust side); this module pins only the *wire*
(``to_dict()`` / ``to_encoding_spec_dict()``) resolution for those, not
rendered output — see the design spec §4.5 and this task's brief.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.axis import Axis, _normalize_axis
from ferrum.configure import AxisConfig
from ferrum.legend import Legend, _normalize_legend
from tests._snapshots import control_chars as _control_chars
from tests._snapshots import legend_texts as _extract_text_labels


def _percent_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})


def _currency_df() -> pl.DataFrame:
    # ``"currency"``'s first character ('c') is itself a valid d3-format type
    # char (Unicode code-point formatting) — an unresolved raw "currency"
    # string is exactly the NF-B1 reproduction: Rust's lenient parser reads
    # 'c' as the type, formats each tick value as `char::from_u32(v)`, and a
    # small tick value like 24/26/28 becomes the literal control character
    # U+0018/U+001A/U+001C. "percent" does not reproduce this (its first
    # char 'p' is also a valid type, but happens to mean percent-format), so
    # the four previously-broken surfaces below use "currency" specifically
    # to discriminate against the real historical bug.
    return pl.DataFrame({"x": [10.0, 20.0, 24.0, 26.0, 28.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})


# ---------------------------------------------------------------------------
# Surface 1 — chart-level AxisConfig (already resolved pre-fix; pinned here
# too so convergence onto the shared resolve_format_field helper doesn't
# regress it).
# ---------------------------------------------------------------------------


def test_axis_config_percent_preset_renders_clean():
    svg = (
        fm.Chart(_percent_df())
        .mark_point()
        .encode(x="x", y="y")
        .configure_axis(label_format="percent")
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"
    labels = _extract_text_labels(svg)
    assert any("%" in label for label in labels), f"no percent-formatted label found: {labels}"


# ---------------------------------------------------------------------------
# Surface 2 — per-channel fm.Axis(label_format=)
# ---------------------------------------------------------------------------


def test_per_channel_axis_currency_preset_renders_clean():
    svg = (
        fm.Chart(_currency_df())
        .mark_point()
        .encode(x=fm.X("x", axis=fm.Axis(label_format="currency")), y="y")
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"
    labels = _extract_text_labels(svg)
    assert any("$" in label for label in labels), f"no currency-formatted label found: {labels}"


def test_axis_to_dict_percent_preset_resolves():
    """Wire-level pin: the preset name itself never reaches to_dict()'s output."""
    d = Axis(label_format="percent").to_dict()
    assert d["label_format"] == ".1%"
    assert d["label_format"] != "percent"
    assert d["label_format_type"] == "number"


# ---------------------------------------------------------------------------
# Surface 3 — per-channel fm.Legend(format=)
# ---------------------------------------------------------------------------


def test_legend_to_dict_percent_preset_resolves():
    d = Legend(format="percent").to_dict()
    assert d["format"] == ".1%"
    assert d["format"] != "percent"
    assert d["format_type"] == "number"


def test_color_legend_currency_preset_renders_clean():
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "c": [10.0, 16.0, 22.0, 28.0],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y", color=fm.Color("c", legend=fm.Legend(format="currency")))
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"


# ---------------------------------------------------------------------------
# Surface 4 — encoding format=
# ---------------------------------------------------------------------------


def test_encoding_format_currency_preset_renders_clean():
    svg = (
        fm.Chart(_currency_df())
        .mark_point()
        .encode(x=fm.X("x", format="currency"), y="y")
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"
    labels = _extract_text_labels(svg)
    assert any("$" in label for label in labels), f"no currency-formatted label found: {labels}"


def test_encoding_format_percent_preset_resolves_on_wire():
    ch = fm.X("x", format="percent")
    d = ch.to_encoding_spec_dict()
    assert d["format"] == ".1%"
    assert d["format"] != "percent"
    assert d["format_type"] == "number"


def test_encoding_explicit_format_type_wins_over_preset_derived():
    """Explicit-format-wins: an explicit format_type= is not overwritten by
    the type a preset would otherwise derive."""
    ch = fm.X("x", format="date_iso", format_type="number")
    d = ch.to_encoding_spec_dict()
    assert d["format_type"] == "number"


# ---------------------------------------------------------------------------
# Surface 5 — raw-dict normalize paths
# ---------------------------------------------------------------------------


def test_normalize_axis_dict_percent_preset_resolves():
    normalized = _normalize_axis({"label_format": "percent"})
    assert normalized["label_format"] == ".1%"
    assert normalized["label_format_type"] == "number"


def test_normalize_legend_dict_percent_preset_resolves():
    normalized = _normalize_legend({"format": "percent"})
    assert normalized["format"] == ".1%"
    assert normalized["format_type"] == "number"


def test_raw_axis_dict_currency_preset_renders_clean():
    svg = (
        fm.Chart(_currency_df())
        .mark_point()
        .encode(x=fm.X("x", axis={"label_format": "currency"}), y="y")
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"
    labels = _extract_text_labels(svg)
    assert any("$" in label for label in labels), f"no currency-formatted label found: {labels}"


def test_raw_axis_dict_camelcase_label_format_renders_clean():
    """axis={'labelFormat': 'currency'} — the camelCase serde-aliased
    spelling (AXIS_STYLE_ALIAS_KEYS in chart_config.rs) — must resolve too,
    not just the snake_case 'label_format' spelling. RED-proved (fix round
    2, quality-reviewer finding on axis.py:327): control chars 0x18/0x1a/
    0x1c reached this SVG before the fix; snake_case on the same chart
    already emitted none."""
    svg = (
        fm.Chart(_currency_df())
        .mark_point()
        .encode(x=fm.X("x", axis={"labelFormat": "currency"}), y="y")
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"
    labels = _extract_text_labels(svg)
    assert any("$" in label for label in labels), f"no currency-formatted label found: {labels}"


def test_normalize_axis_dict_camelcase_writes_back_camelcase_spelling():
    """Resolution writes the resolved spec/type back under the caller's own
    spelling — camelCase in, camelCase out, never a second snake_case
    spelling of the same field alongside it."""
    normalized = _normalize_axis({"labelFormat": "percent"})
    assert normalized["labelFormat"] == ".1%"
    assert normalized["labelFormatType"] == "number"
    assert "label_format" not in normalized
    assert "label_format_type" not in normalized


def test_normalize_axis_dict_mixed_spelling_no_duplicate_field_key():
    """A raw dict combining snake_case 'label_format' with camelCase
    'labelFormatType' must not have a second spelling of the type key added
    beside the caller-supplied one — previously this combination raised an
    untyped serde 'duplicate field `label_format_type`' error when forwarded
    to Rust (fix round 2, quality-reviewer finding on axis.py:337)."""
    normalized = _normalize_axis({"label_format": "percent", "labelFormatType": "time"})
    assert normalized["label_format"] == ".1%"
    # Explicit format_type always wins over the preset-derived one.
    assert normalized["labelFormatType"] == "time"
    assert "label_format_type" not in normalized

    # Render end to end to prove the previously-crashing combination no
    # longer reaches Rust as a duplicate-field wire error.
    svg = (
        fm.Chart(_percent_df())
        .mark_point()
        .encode(x=fm.X("x", axis={"label_format": "percent", "labelFormatType": "number"}), y="y")
        .to_svg()
    )
    assert not _control_chars(svg)


def test_normalize_axis_dict_copies_unconditionally():
    """_normalize_axis's dict path must never return the caller's own dict
    object, regardless of whether a format key is present — one aliasing
    contract, not one that depends on the dict's contents (fix round 2,
    quality-reviewer finding on axis.py:358)."""
    no_format = {"grid": False}
    assert _normalize_axis(no_format) is not no_format

    with_format = {"label_format": "percent"}
    assert _normalize_axis(with_format) is not with_format


def test_raw_legend_dict_currency_preset_renders_clean():
    """Render-level pin for the raw-dict LEGEND surface (fm.Color(legend={...})).

    Previously pinned at the wire level only
    (test_normalize_legend_dict_percent_preset_resolves); the spec review's
    non-blocking observation flagged this as the one of five surfaces
    without a rendered-output pin. Uses "currency" (not "percent") for the
    same discriminating reason as the other render-level pins above."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "c": [10.0, 16.0, 22.0, 28.0],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y", color=fm.Color("c", legend={"format": "currency"}))
        .to_svg()
    )
    bad = _control_chars(svg)
    assert not bad, f"control chars in SVG: {[hex(ord(c)) for c in bad]}"


def test_normalize_legend_dict_copies_unconditionally():
    """_normalize_legend's dict path must never return the caller's own
    dict object, regardless of whether a format key is present — same
    aliasing contract as _normalize_axis (fix round 2, quality-reviewer
    finding on legend.py:196)."""
    no_format = {"orient": "top"}
    assert _normalize_legend(no_format) is not no_format

    with_format = {"format": "percent"}
    assert _normalize_legend(with_format) is not with_format


# ---------------------------------------------------------------------------
# Unknown-name-passes-raw (cross-surface pin, NF-B1)
# ---------------------------------------------------------------------------


def test_axis_config_unrecognized_format_name_raises_at_construction():
    """AxisConfig.label_format is preset-names-only by contract — it is not
    one of the four raw-spec-accepting surfaces (that's the dedicated,
    mutually-exclusive label_format_raw sibling). An unrecognized name (a
    typo, e.g. "curency") is a typed ValueError at construction, not a
    silent raw pass-through: the pass-through path is exactly what let the
    literal control-character bytes (NF-B1) reach the SVG on this surface."""
    with pytest.raises(ValueError, match="Unknown format preset"):
        AxisConfig(label_format="curency")


def test_configure_axis_unrecognized_format_name_raises_at_construction():
    """Same pin via the chainable Chart.configure_axis() entry point."""
    with pytest.raises(ValueError, match="Unknown format preset"):
        fm.Chart(_percent_df()).mark_point().encode(x="x", y="y").configure_axis(
            label_format="curency"
        )


def test_axis_unknown_format_name_passes_through_raw():
    d = Axis(label_format="not_a_real_preset").to_dict()
    assert d["label_format"] == "not_a_real_preset"


def test_legend_unknown_format_name_passes_through_raw():
    d = Legend(format="not_a_real_preset").to_dict()
    assert d["format"] == "not_a_real_preset"


def test_encoding_unknown_format_name_passes_through_raw():
    d = fm.X("x", format="not_a_real_preset").to_encoding_spec_dict()
    assert d["format"] == "not_a_real_preset"


def test_normalize_axis_dict_unknown_format_name_passes_through_raw():
    normalized = _normalize_axis({"label_format": "not_a_real_preset"})
    assert normalized["label_format"] == "not_a_real_preset"


def test_normalize_legend_dict_unknown_format_name_passes_through_raw():
    normalized = _normalize_legend({"format": "not_a_real_preset"})
    assert normalized["format"] == "not_a_real_preset"


# ---------------------------------------------------------------------------
# Time presets — wire-level resolution only. Rendered date formatting at the
# axis/legend level requires format_type consumption that lands in batch B
# Task 4; this task pins what resolves at the wire, not rendered output.
# ---------------------------------------------------------------------------


def test_axis_config_time_preset_resolves_on_wire():
    d = AxisConfig(label_format="date_iso").to_dict()
    assert d["label_format"] == "%Y-%m-%d"
    assert d["label_format_type"] == "time"


def test_axis_time_preset_resolves_on_wire():
    d = Axis(label_format="date_iso").to_dict()
    assert d["label_format"] == "%Y-%m-%d"
    assert d["label_format_type"] == "time"


def test_axis_explicit_format_type_wins_over_preset_derived():
    d = Axis(label_format="date_iso", label_format_type="number").to_dict()
    assert d["label_format_type"] == "number"


def test_legend_time_preset_resolves_on_wire():
    d = Legend(format="date_iso").to_dict()
    assert d["format"] == "%Y-%m-%d"
    assert d["format_type"] == "time"


def test_legend_explicit_format_type_wins_over_preset_derived():
    d = Legend(format="date_iso", format_type="number").to_dict()
    assert d["format_type"] == "number"


def test_encoding_format_time_preset_resolves_on_wire():
    d = fm.X("x", format="date_iso").to_encoding_spec_dict()
    assert d["format"] == "%Y-%m-%d"
    assert d["format_type"] == "time"


def test_normalize_axis_dict_time_preset_resolves_on_wire():
    normalized = _normalize_axis({"label_format": "date_iso"})
    assert normalized["label_format"] == "%Y-%m-%d"
    assert normalized["label_format_type"] == "time"


def test_normalize_legend_dict_time_preset_resolves_on_wire():
    normalized = _normalize_legend({"format": "date_iso"})
    assert normalized["format"] == "%Y-%m-%d"
    assert normalized["format_type"] == "time"


def test_axis_ordinal_preset_resolves_to_sentinel_on_wire():
    """'ordinal' resolves to the __ordinal__ sentinel; suffix rendering is
    Task 4's Rust-side format.rs work (D8)."""
    d = Axis(label_format="ordinal").to_dict()
    assert d["label_format"] == "__ordinal__"
    assert d["label_format_type"] == "number"


# ---------------------------------------------------------------------------
# resolve_format_or_raw / resolve_format_field — direct unit coverage
# ---------------------------------------------------------------------------


class TestResolveFormatOrRaw:
    def test_known_numeric_preset_resolves_with_type(self):
        from ferrum.format_presets import resolve_format_or_raw

        assert resolve_format_or_raw("percent") == (".1%", "number")

    def test_known_time_preset_resolves_with_type(self):
        from ferrum.format_presets import resolve_format_or_raw

        assert resolve_format_or_raw("date_iso") == ("%Y-%m-%d", "time")

    def test_unknown_name_passes_through_with_no_type(self):
        from ferrum.format_presets import resolve_format_or_raw

        assert resolve_format_or_raw(",.2f") == (",.2f", None)
        assert resolve_format_or_raw("not_a_preset") == ("not_a_preset", None)

    def test_non_str_value_passes_through_without_raising(self):
        """A non-hashable value (e.g. a list) must not raise a bare TypeError
        from the `in NUMERIC_PRESETS` membership test — it passes through
        unchanged so the caller's own typed refusal fires instead (fix
        round 2, quality-reviewer finding on format_presets.py:117)."""
        from ferrum.format_presets import resolve_format_or_raw

        unhashable = ["not", "a", "string"]
        assert resolve_format_or_raw(unhashable) == (unhashable, None)
        assert resolve_format_or_raw(42) == (42, None)

    @pytest.mark.parametrize("name", ["integer", "currency", "si", "compact", "scientific"])
    def test_all_numeric_presets_resolve_to_number_type(self, name):
        from ferrum.format_presets import resolve_format_or_raw

        _, format_type = resolve_format_or_raw(name)
        assert format_type == "number"

    @pytest.mark.parametrize("name", ["date_short", "month", "year", "time_12h", "datetime"])
    def test_all_time_presets_resolve_to_time_type(self, name):
        from ferrum.format_presets import resolve_format_or_raw

        _, format_type = resolve_format_or_raw(name)
        assert format_type == "time"


class TestResolveFormatField:
    def test_none_raw_value_resolves_to_none_pair_minus_explicit_type(self):
        from ferrum.format_presets import resolve_format_field

        assert resolve_format_field(None, None) == (None, None)
        assert resolve_format_field(None, "time") == (None, "time")

    def test_explicit_type_wins_over_preset_derived(self):
        from ferrum.format_presets import resolve_format_field

        spec, format_type = resolve_format_field("date_iso", "number")
        assert spec == "%Y-%m-%d"
        assert format_type == "number"

    def test_unset_type_falls_back_to_preset_derived(self):
        from ferrum.format_presets import resolve_format_field

        spec, format_type = resolve_format_field("date_iso", None)
        assert spec == "%Y-%m-%d"
        assert format_type == "time"
