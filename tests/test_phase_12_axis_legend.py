"""Tests for Phase 12 Axis and Legend value classes.

Two layers of coverage live here:

1. ``.to_dict()`` serialization contract (``TestAxisToDict`` / ``TestLegendToDict``
   and the normalize tests). These prove the Python dataclasses serialize their
   fields. They are necessary but *not sufficient* — historically (B5 RCA,
   ``design-docs/superpowers/followups/2026-06-14-per-channel-axis-legend-silent-drop-rca.md``)
   a feature shipped where every per-channel ``fm.Axis``/``fm.Legend`` field
   serialized cleanly into the spec and was then silently dropped at render,
   yet these ``.to_dict()`` assertions stayed green.

2. Render-level coverage (``TestAxisRender`` / ``TestLegendRender`` and the
   parity / fail-loud / golden-stability classes). Each previously-dropped
   per-channel field is asserted on the *parsed SVG* — the observable effect a
   user would see — closing the test-quality gap the RCA names.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import polars as pl
import pytest

import ferrum as fm
from ferrum.axis import Axis, _normalize_axis, _axis_suppressed_dict
from ferrum.legend import Legend, _normalize_legend


_SVG_NS = "{http://www.w3.org/2000/svg}"


def _svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _lines(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//" + _SVG_NS + "line")


def _texts(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//" + _SVG_NS + "text")


def _x_grid_lines(root: ET.Element) -> list[ET.Element]:
    """Vertical grid lines (x1 == x2) spanning the plot — the x-axis grid."""
    out: list[ET.Element] = []
    for ln in _lines(root):
        x1, x2 = ln.get("x1"), ln.get("x2")
        if x1 is not None and x1 == x2:
            out.append(ln)
    return out


def _x_domain_line(root: ET.Element) -> ET.Element | None:
    """The bottom x-axis domain line: a wide, near-horizontal line at the panel base.

    The domain line runs left-to-right at the bottom of the panel, so its x1/x2
    differ while y1/y2 are close. Several horizontal grid lines share its width,
    so it is disambiguated as the *lowest* (largest y1) such line — the panel
    floor where the x axis sits.
    """
    candidates: list[tuple[float, ET.Element]] = []
    for ln in _lines(root):
        try:
            x1 = float(ln.get("x1", "nan"))
            x2 = float(ln.get("x2", "nan"))
            y1 = float(ln.get("y1", "nan"))
            y2 = float(ln.get("y2", "nan"))
        except ValueError:
            continue
        if abs(y1 - y2) <= 2.0 and (x2 - x1) > 100.0 and y1 > 100.0:
            candidates.append((y1, ln))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]


def _x_tick_label_anchors(svg: str) -> list[str]:
    """Text-anchors of the x-axis tick labels, ordered left-to-right.

    The x-axis labels sit below the plot (largest ``y``) and are the numeric
    tick texts; y-axis labels sit to the left and default to ``end`` anchoring.
    We isolate the x labels as the numeric texts at the maximal label ``y`` band,
    then sort by ``x`` so index 0 is the first (leftmost) tick label and -1 the
    last. This makes the flush boundary anchors (start on the first, end on the
    last) directly assertable without the y labels interfering.
    """
    root = _svg_root(svg)
    labels: list[tuple[float, float, str]] = []
    for t in _texts(root):
        content = (t.text or "").strip()
        if not re.fullmatch(r"[0-9.]+", content):
            continue
        try:
            x = float(t.get("x", "nan"))
            y = float(t.get("y", "nan"))
        except ValueError:
            continue
        anchor = t.get("text-anchor", "middle")
        labels.append((x, y, anchor))
    if not labels:
        return []
    max_y = max(y for _, y, _ in labels)
    x_band = [(x, anchor) for x, y, anchor in labels if abs(y - max_y) <= 2.0]
    return [anchor for _, anchor in sorted(x_band, key=lambda p: p[0])]


@pytest.fixture()
def scatter_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": list(range(10)),
        }
    )


@pytest.fixture()
def color_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "g": ["a", "b", "c", "d"],
        }
    )


# ---------------------------------------------------------------------------
# Axis tests
# ---------------------------------------------------------------------------


class TestAxisFrozen:
    """Axis instances are immutable (frozen dataclass)."""

    def test_frozen(self):
        ax = Axis(title="Speed")
        with pytest.raises(AttributeError):
            ax.title = "Other"  # type: ignore[misc]

    def test_slots(self):
        ax = Axis()
        with pytest.raises((AttributeError, TypeError)):
            ax.new_attr = 42  # type: ignore[attr-defined]


class TestAxisToDict:
    """Axis.to_dict() serializes correctly, omitting defaults and None."""

    def test_empty(self):
        # All defaults — empty dict
        assert Axis().to_dict() == {}

    def test_title_and_grid_false(self):
        result = Axis(title="Speed", grid=False).to_dict()
        assert result == {"title": "Speed", "grid": False}

    def test_all_bool_suppressed(self):
        result = Axis(ticks=False, labels=False, domain=False, grid=False).to_dict()
        assert result == {"ticks": False, "labels": False, "domain": False, "grid": False}

    def test_tick_count(self):
        result = Axis(tick_count=5).to_dict()
        assert result == {"tick_count": 5}

    def test_grid_styling(self):
        result = Axis(grid_dash=[4.0, 2.0], grid_width=0.5, grid_color="#ccc").to_dict()
        assert result == {"grid_dash": [4.0, 2.0], "grid_width": 0.5, "grid_color": "#ccc"}

    def test_label_options(self):
        # "greedy" is the default overlap strategy and is omitted; "parity" is a
        # non-default value and must round-trip.
        result = Axis(label_angle=45.0, label_overlap="parity").to_dict()
        assert result == {"label_angle": 45.0, "label_overlap": "parity"}

    def test_label_overlap_default_omitted(self):
        # Omitting label_overlap entirely omits the key (renderer default applies).
        # NF-B3: an EXPLICIT "greedy" is no longer indistinguishable from "not
        # specified" — it always reaches the wire even though it textually
        # matches the renderer's own default. See
        # TestAxisRender.test_explicit_label_overlap_beats_chart_level for the
        # observable divergence this fixes.
        assert Axis(label_overlap="greedy").to_dict() == {"label_overlap": "greedy"}
        assert Axis().to_dict() == {}

    def test_label_flush_explicit_not_omitted(self):
        # label_flush's renderer default is False; an explicit False now reaches
        # the wire too (NF-B3), not just an explicit True — only OMITTING the
        # field drops the key.
        assert Axis(label_flush=False).to_dict() == {"label_flush": False}
        assert Axis(label_flush=True).to_dict() == {"label_flush": True}
        assert Axis().to_dict() == {}

    @pytest.mark.parametrize(
        "field", ["ticks", "tick_extra", "grid", "labels", "label_flush", "label_overlap", "domain"]
    )
    def test_explicit_none_is_unspecified_for_every_unset_defaulted_field(self, field):
        # Quality-review fix (S2, cycle 3): Axis's sentinel-skip block used to
        # gate on `is not _UNSET` only, so an explicit None on one of these
        # seven fields bypassed the omission and reached to_dict() as a raw
        # {"field": None} entry -- inert in Rust (Option<...> fields treat
        # null like absent) but inconsistent with every other field's None
        # convention. ferrum._title_sentinel.is_unspecified now gates all
        # seven the same way _UNSET already did, so explicit None is dropped
        # exactly like omitting the kwarg.
        assert Axis(**{field: None}).to_dict() == {}

    def test_orient(self):
        result = Axis(orient="top").to_dict()
        assert result == {"orient": "top"}

    def test_values_explicit(self):
        result = Axis(values=[0, 25, 50, 75, 100]).to_dict()
        assert result == {"values": [0, 25, 50, 75, 100]}


class TestAxisNormalize:
    """_normalize_axis handles all input forms."""

    def test_none_passthrough(self):
        assert _normalize_axis(None) is None

    def test_false_suppression(self):
        result = _normalize_axis(False)
        assert result == _axis_suppressed_dict()
        assert result["ticks"] is False
        assert result["labels"] is False
        assert result["domain"] is False
        assert result["grid"] is False

    def test_axis_instance(self):
        ax = Axis(title="X", grid=False)
        result = _normalize_axis(ax)
        assert result == {"title": "X", "grid": False}

    def test_dict_passthrough(self):
        # NF-B1 fix round (2026-09-02): _normalize_axis's dict path now
        # copies unconditionally (one aliasing contract regardless of
        # whether a label_format key is present — quality-reviewer finding
        # on axis.py:358), so it no longer returns the caller's own dict
        # object. Content still passes through unchanged when there is no
        # label_format key to resolve.
        d = {"title": "Custom", "ticks": False}
        result = _normalize_axis(d)
        assert result == d
        assert result is not d


# ---------------------------------------------------------------------------
# Legend tests
# ---------------------------------------------------------------------------


class TestLegendFrozen:
    """Legend instances are immutable (frozen dataclass)."""

    def test_frozen(self):
        lg = Legend(title="Species")
        with pytest.raises(AttributeError):
            lg.title = "Other"  # type: ignore[misc]

    def test_slots(self):
        lg = Legend()
        with pytest.raises((AttributeError, TypeError)):
            lg.new_attr = 42  # type: ignore[attr-defined]


class TestLegendToDict:
    """Legend.to_dict() serializes correctly, omitting only unset/None values."""

    def test_empty(self):
        # orient/direction are omitted only when NOT PASSED at all (the
        # renderer's own default then applies).
        assert Legend().to_dict() == {}

    def test_orient_bottom_columns(self):
        result = Legend(orient="bottom", columns=3).to_dict()
        assert result == {"orient": "bottom", "columns": 3}

    def test_direction_horizontal(self):
        result = Legend(direction="horizontal").to_dict()
        assert result == {"direction": "horizontal"}

    def test_orient_direction_explicit_equals_default_still_serializes(self):
        # F-L04-04/NF-B3: fm.Legend(direction="vertical") used to strip to {}
        # because "vertical" textually matched the Python default, making an
        # explicit value indistinguishable from "not specified" and silently
        # losing the per-channel-wins cascade for it. Both previously-dead
        # combinations from the fm.Legend surface now reach the wire.
        assert Legend(direction="vertical").to_dict() == {"direction": "vertical"}
        assert Legend(orient="right", direction="vertical").to_dict() == {
            "orient": "right",
            "direction": "vertical",
        }

    def test_orient_none_reaches_wire_verbatim(self):
        # Per-channel orient="none" is a suppression Rust consumes directly
        # (LegendStyleSpec::suppresses()) — it must not be normalized away in
        # Python the way chart-level orient="none" is.
        assert Legend(orient="none").to_dict() == {"orient": "none"}

    def test_invalid_orient_and_direction_are_refused(self):
        with pytest.raises(ValueError, match="orient"):
            Legend(orient="diagonal")
        with pytest.raises(ValueError, match="direction"):
            Legend(direction="sideways")

    def test_explicit_none_is_unspecified_not_a_refusal(self):
        # Quality-review fix (S2, cycle 3): Legend.__post_init__'s validator
        # gate used to be `is not _UNSET`, so an explicit orient=None/
        # direction=None (Python's universal "unset" spelling for every
        # OTHER optional field on this class) fell through to
        # validate_choice and RAISED ("orient must be one of [...]; got
        # None") -- a regression from pre-batch behavior, where None was
        # silently dropped like any other unset field. is_unspecified now
        # treats None the same as _UNSET on this surface, matching
        # LegendConfig's pre-existing `orient is not None` convention (see
        # tests/test_configure.py::TestLegendConfig).
        assert Legend(orient=None).to_dict() == {}
        assert Legend(direction=None).to_dict() == {}

    def test_none_semantics_agree_across_all_three_legend_surfaces(self):
        # RED-proof / divergence pin (S2, cycle 3): before this fix, the same
        # None token was handled three incompatible ways across the three
        # Python surfaces the shared orient/direction validator was
        # introduced to unify:
        #   - Legend(orient=None)            -> RAISED ValueError
        #   - LegendConfig(orient=None)       -> accepted as "unset" (no raise)
        #   - a raw {"orient": None} dict     -> RAISED ValueError (via
        #     _normalize_legend, same gate as Legend.__post_init__)
        # All three now agree: None is "not specified", never a refusal.
        from ferrum.configure import LegendConfig
        from ferrum.legend import _normalize_legend

        assert Legend(orient=None).to_dict() == {}
        assert LegendConfig(orient=None).to_dict() == {}
        normalized = _normalize_legend({"orient": None})
        # The raw-dict path passes keys through unfiltered (unrelated to this
        # fix -- dicts never strip None for ANY field), so `orient: None`
        # survives in the dict; what changed is that building it no longer
        # raises. A JSON `null` there is indistinguishable from an absent key
        # to Rust's Option<...> deserializer, so this is not a wire regression.
        assert normalized is not None and normalized.get("orient") is None

    def test_symbol_options(self):
        result = Legend(symbol_size=100.0, symbol_type="square").to_dict()
        assert result == {"symbol_size": 100.0, "symbol_type": "square"}

    def test_gradient_options(self):
        result = Legend(type="gradient", gradient_length=200.0).to_dict()
        assert result == {"type": "gradient", "gradient_length": 200.0}

    def test_title_and_font(self):
        result = Legend(title="Category", title_font_size=14.0, label_font_size=11.0).to_dict()
        assert result == {
            "title": "Category",
            "title_font_size": 14.0,
            "label_font_size": 11.0,
        }


class TestLegendNormalize:
    """_normalize_legend handles all input forms."""

    def test_none_suppression(self):
        result = _normalize_legend(None)
        assert result == {"disabled": True}

    def test_false_suppression(self):
        result = _normalize_legend(False)
        assert result == {"disabled": True}

    def test_legend_instance(self):
        lg = Legend(orient="bottom", columns=2)
        result = _normalize_legend(lg)
        assert result == {"orient": "bottom", "columns": 2}

    def test_dict_passthrough(self):
        # NF-B1 fix round (2026-09-02): _normalize_legend's dict path now
        # copies unconditionally (one aliasing contract regardless of
        # whether a format key is present — quality-reviewer finding on
        # legend.py:196), so it no longer returns the caller's own dict
        # object. Content still passes through unchanged when there is no
        # format key to resolve.
        d = {"orient": "left", "title": "Size"}
        result = _normalize_legend(d)
        assert result == d
        assert result is not d

    def test_other_truthy_reserved(self):
        # Unknown truthy values -> None (reserved)
        assert _normalize_legend(True) is None
        assert _normalize_legend(42) is None


# ---------------------------------------------------------------------------
# Integration with encoding channels
# ---------------------------------------------------------------------------


class TestAxisEncodingIntegration:
    """Axis value class integrates with encoding channels."""

    def test_axis_false_on_x(self):
        enc = fm.X("field", axis=False)
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == _axis_suppressed_dict()

    def test_axis_instance_on_x(self):
        enc = fm.X("field", axis=Axis(title="X Axis", grid=False))
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == {"title": "X Axis", "grid": False}

    def test_axis_dict_on_x(self):
        """Backward compat: axis as raw dict still works."""
        enc = fm.X("field", axis={"title": "X"})
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == {"title": "X"}

    def test_axis_instance_on_y(self):
        enc = fm.Y("mpg", axis=Axis(orient="right", tick_count=10))
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == {"orient": "right", "tick_count": 10}

    def test_axis_none_not_set(self):
        """axis=None means 'not specified' — no axis key in output."""
        enc = fm.X("field")
        d = enc.to_encoding_spec_dict()
        assert "axis" not in d


class TestLegendEncodingIntegration:
    """Legend value class integrates with encoding channels."""

    def test_legend_none_suppresses(self):
        enc = fm.Color("species", legend=None)
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"disabled": True}

    def test_legend_false_suppresses(self):
        enc = fm.Color("species", legend=False)
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"disabled": True}

    def test_legend_instance_on_color(self):
        enc = fm.Color("species", legend=Legend(orient="bottom"))
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"orient": "bottom"}

    def test_legend_dict_on_color(self):
        """Backward compat: legend as raw dict still works."""
        enc = fm.Color("species", legend={"orient": "left"})
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"orient": "left"}

    def test_legend_instance_with_columns(self):
        enc = fm.Color("origin", legend=Legend(orient="bottom", columns=3, direction="horizontal"))
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"orient": "bottom", "columns": 3, "direction": "horizontal"}


# ---------------------------------------------------------------------------
# Public API availability
# ---------------------------------------------------------------------------


class TestPublicExports:
    """Axis and Legend are importable from ferrum top level."""

    def test_axis_importable(self):
        assert fm.Axis is Axis

    def test_legend_importable(self):
        assert fm.Legend is Legend


# ---------------------------------------------------------------------------
# Render-level coverage — per-channel fm.Axis(...) fields
#
# Each test sets a single field per-channel and asserts its observable effect on
# the parsed SVG. This is the coverage the B5 RCA §5 found missing: the fields
# below all serialized via .to_dict() (tested above) but were silently dropped
# at render until B5 units 1-4 wired them through the typed per-channel path.
# ---------------------------------------------------------------------------


class TestAxisRender:
    """Per-channel ``fm.Axis`` fields reach the rendered SVG."""

    def test_grid_color(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(grid_color="#abc123")), y="y")
            .to_svg()
        )
        styled = [ln for ln in _x_grid_lines(_svg_root(svg)) if ln.get("stroke") == "#abc123"]
        assert styled, "per-channel grid_color must color the x grid lines"

    def test_grid_dash(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(grid_dash=[4.0, 2.0])), y="y")
            .to_svg()
        )
        dashed = [ln for ln in _x_grid_lines(_svg_root(svg)) if ln.get("stroke-dasharray") == "4,2"]
        assert dashed, "per-channel grid_dash must set stroke-dasharray on x grid lines"

    def test_grid_width(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(grid_width=3.0)), y="y")
            .to_svg()
        )
        wide = [ln for ln in _x_grid_lines(_svg_root(svg)) if ln.get("stroke-width") == "3"]
        assert wide, "per-channel grid_width must set stroke-width on x grid lines"

    def test_grid_opacity(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(grid_opacity=0.3)), y="y")
            .to_svg()
        )
        faint = [ln for ln in _x_grid_lines(_svg_root(svg)) if ln.get("stroke-opacity") == "0.3"]
        assert faint, "per-channel grid_opacity must set stroke-opacity on x grid lines"

    def test_domain_color_and_width(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(domain_color="#123456", domain_width=4.0)), y="y")
            .to_svg()
        )
        domain = _x_domain_line(_svg_root(svg))
        assert domain is not None, "expected an x-axis domain line"
        assert domain.get("stroke") == "#123456", "per-channel domain_color must color domain line"
        assert domain.get("stroke-width") == "4", "per-channel domain_width must size domain line"

    def test_label_color(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_color="#ff00ff")), y="y")
            .to_svg()
        )
        magenta = [
            t
            for t in _texts(_svg_root(svg))
            if t.get("fill") == "#ff00ff" and t.get("text-anchor") == "middle"
        ]
        assert magenta, "per-channel label_color must color x tick labels magenta"

    def test_label_angle_rotates(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_angle=-30)), y="y")
            .to_svg()
        )
        rotated = [t for t in _texts(_svg_root(svg)) if "rotate(-30" in (t.get("transform") or "")]
        assert rotated, "per-channel label_angle must rotate x tick labels by -30"

    def test_label_font_size_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_font_size=20.0)), y="y")
            .to_svg()
        )
        assert 'font-size="20"' in svg, "per-channel label_font_size must size tick labels"
        assert svg != base, "per-channel label_font_size must change the SVG"

    def test_title_color(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(title="Speed", title_color="#aa0000")), y="y")
            .to_svg()
        )
        titled = [
            t
            for t in _texts(_svg_root(svg))
            if (t.text or "").strip() == "Speed" and t.get("fill") == "#aa0000"
        ]
        assert titled, "per-channel title_color must color the axis title"

    def test_title_font_size(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(title="Speed", title_font_size=22.0)), y="y")
            .to_svg()
        )
        sized = [
            t
            for t in _texts(_svg_root(svg))
            if (t.text or "").strip() == "Speed" and t.get("font-size") == "22"
        ]
        assert sized, "per-channel title_font_size must size the axis title"

    def test_title_padding_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(title="Speed")), y="y")
            .to_svg()
        )
        padded = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(title="Speed", title_padding=40.0)), y="y")
            .to_svg()
        )
        assert padded != base, "per-channel title_padding must change axis-title placement"

    def test_title_orient_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(title="X")),
                y=fm.Y("y", axis=fm.Axis(title="Y")),
            )
            .to_svg()
        )
        oriented = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(title="X", title_orient="top")),
                y=fm.Y("y", axis=fm.Axis(title="Y", title_orient="top")),
            )
            .to_svg()
        )
        assert oriented != base, "per-channel title_orient must reposition axis titles"

    def test_tick_count(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        reduced = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(tick_count=4)), y="y")
            .to_svg()
        )
        assert reduced != base, "per-channel tick_count must change the tick set"
        # Fewer x tick labels than the 10-value default.
        labels = re.findall(r'text-anchor="middle"[^>]*>([0-9.]+)</text>', reduced)
        assert 0 < len(labels) < 10, f"tick_count=4 should thin the x tick labels, got {labels}"

    def test_tick_min_step(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        stepped = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(tick_min_step=4.0)), y="y")
            .to_svg()
        )
        assert stepped != base, "per-channel tick_min_step must thin the tick set"

    def test_values_explicit(self, scatter_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(values=[2.0, 5.0, 8.0])), y="y")
            .to_svg()
        )
        labels = re.findall(r'text-anchor="middle"[^>]*>([0-9.]+)</text>', svg)
        assert labels == ["2", "5", "8"], (
            f"per-channel values=[2,5,8] must place ticks at exactly those values, got {labels}"
        )

    def test_orient_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        top = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(orient="top")), y="y")
            .to_svg()
        )
        assert top != base, "per-channel orient='top' must move the x axis to the top"

    def test_translate_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        shifted = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(translate=15.0)), y="y")
            .to_svg()
        )
        assert shifted != base, "per-channel translate must shift the axis"

    def test_min_band_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        extended = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y=fm.Y("y", axis=fm.Axis(min_band=120.0)))
            .to_svg()
        )
        assert extended != base, "per-channel min_band must change axis-margin layout"

    def test_max_band_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        clamped = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y=fm.Y("y", axis=fm.Axis(max_band=20.0)))
            .to_svg()
        )
        assert clamped != base, "per-channel max_band must change axis-margin layout"

    def test_zindex_changes_render(self, scatter_df: pl.DataFrame) -> None:
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        layered = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(zindex=1)), y="y")
            .to_svg()
        )
        assert layered != base, "per-channel zindex must change grid/mark layering"

    def test_label_flush_anchors_boundary_labels(self, scatter_df: pl.DataFrame) -> None:
        # The default (flush off) anchors every x tick label "middle"; flush=True
        # left-anchors the first label and right-anchors the last so the boundary
        # labels do not overhang the plot edges. Before this fix the True value was
        # silently dropped (it matched the Python default) and never reached Rust.
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        flushed = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_flush=True)), y="y")
            .to_svg()
        )
        assert flushed != base, "label_flush=True must change the x tick-label anchoring"
        base_anchors = _x_tick_label_anchors(base)
        flush_anchors = _x_tick_label_anchors(flushed)
        assert base_anchors and set(base_anchors) == {"middle"}, (
            f"default (flush off) must keep every x tick label middle-anchored, got {base_anchors}"
        )
        assert flush_anchors[0] == "start", (
            f"label_flush=True must left-anchor the first x tick label, got {flush_anchors}"
        )
        assert flush_anchors[-1] == "end", (
            f"label_flush=True must right-anchor the last x tick label, got {flush_anchors}"
        )

    def test_label_flush_false_matches_default(self, scatter_df: pl.DataFrame) -> None:
        # label_flush=False is the renderer default, so it must be byte-identical
        # to omitting the field entirely (no silent render change either way).
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        explicit = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_flush=False)), y="y")
            .to_svg()
        )
        assert explicit == base, "label_flush=False must render identically to the default"

    def test_label_overlap_parity_thins_labels(self) -> None:
        # A dense integer x axis where the greedy cascade keeps most labels; parity
        # culls to a stride-2 subset, so it must show strictly fewer x tick labels.
        df = pl.DataFrame({"x": list(range(20)), "y": list(range(20))})
        greedy = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_overlap="greedy")), y="y")
            .to_svg()
        )
        parity = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_overlap="parity")), y="y")
            .to_svg()
        )
        assert parity != greedy, "label_overlap='parity' must change the rendered tick set"

        def _x_label_count(svg: str) -> int:
            return len(re.findall(r'text-anchor="middle"[^>]*>\d+</text>', svg))

        assert _x_label_count(parity) < _x_label_count(greedy), (
            "label_overlap='parity' must show fewer x tick labels than 'greedy'"
        )

    def test_label_overlap_greedy_matches_default(self, scatter_df: pl.DataFrame) -> None:
        # "greedy" is the renderer's own default, so an explicit "greedy" (which
        # NF-B3 now puts on the wire, see TestAxisToDict) still renders
        # identically to omitting the field — the value Rust resolves to is the
        # same either way, only its PROVENANCE (explicit vs. defaulted) differs,
        # which is what test_explicit_label_overlap_beats_chart_level below
        # exercises.
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        explicit = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_overlap="greedy")), y="y")
            .to_svg()
        )
        assert explicit == base, "label_overlap='greedy' must render identically to the default"

    def test_explicit_label_overlap_beats_chart_level(self, scatter_df: pl.DataFrame) -> None:
        # NF-B3's real divergence: an EXPLICIT per-channel label_overlap="greedy"
        # (textually equal to the Python default) must now beat a conflicting
        # chart-level configure_axis(label_overlap=...), because per-channel
        # always wins over chart-level. Before the _AXIS_DEFAULTS fix, the
        # explicit value was silently stripped on the way to the wire and was
        # therefore indistinguishable from "not specified", so chart-level
        # filled in and this pair rendered identically (a real bug: explicit
        # per-channel intent was lost). Rust's cascade already resolves this
        # correctly on x (render/mod.rs::chart_config_offset_flush_overlap_do_not_override_per_channel);
        # this pins that the Python wire-emission half is no longer the blocker.
        explicit_wins = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_overlap="greedy")), y="y")
            .configure_axis(label_overlap="parity")
            .to_svg()
        )
        chart_level_only = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(label_overlap="parity")
            .to_svg()
        )
        assert explicit_wins != chart_level_only, (
            "an explicit per-channel label_overlap='greedy' (even though it equals "
            "the Python default) must beat a conflicting chart-level "
            "configure_axis(label_overlap='parity')"
        )


# ---------------------------------------------------------------------------
# Render-level coverage — per-channel fm.Legend(...) fields
# ---------------------------------------------------------------------------


class TestLegendRender:
    """Per-channel ``fm.Legend`` fields reach the rendered SVG."""

    def test_symbol_type_square(self, color_df: pl.DataFrame) -> None:
        """symbol_type='square' renders rect swatches instead of circle swatches."""
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        squared = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(symbol_type="square")))
            .to_svg()
        )
        assert squared != base, "per-channel symbol_type must change the swatch shape"
        # A square swatch is a small (8x8) rect near the legend on the right edge.
        root = _svg_root(squared)
        square_swatches = [
            r
            for r in root.findall(".//" + _SVG_NS + "rect")
            if r.get("width") == "8" and r.get("height") == "8" and float(r.get("x", "0")) > 500
        ]
        assert square_swatches, "symbol_type='square' must render square legend swatches"

    def test_symbol_stroke_width(self, color_df: pl.DataFrame) -> None:
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        thick = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(symbol_stroke_width=3.0)))
            .to_svg()
        )
        assert thick != base, "per-channel symbol_stroke_width must change the SVG"
        assert 'stroke-width="3"' in thick, "symbol_stroke_width=3 must appear on swatches"

    def test_label_limit_truncates_with_ellipsis(self) -> None:
        long_label = "an extremely long category label that overflows badly"
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [10, 20, 30, 40],
                "g": [long_label, "short", long_label, "short"],
            }
        )
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        limited = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(label_limit=40.0)))
            .to_svg()
        )
        assert limited != base, "per-channel label_limit must change the SVG"
        assert "…" in limited, "per-channel label_limit must truncate with an ellipsis (…)"

    def test_clip_height_adds_clip_path(self, color_df: pl.DataFrame) -> None:
        clipped = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(clip_height=40.0)))
            .to_svg()
        )
        clip_ids = re.findall(r'<clipPath id="([^"]+)"', clipped)
        assert any("legend-clip" in cid for cid in clip_ids), (
            f"per-channel clip_height must add a legend clipPath, got {clip_ids}"
        )

    def test_row_padding_spaces_entries(self, color_df: pl.DataFrame) -> None:
        def legend_label_ys(svg: str) -> list[float]:
            return [
                float(t.get("y", "nan"))
                for t in _texts(_svg_root(svg))
                if (t.text or "").strip() in {"a", "b", "c", "d"}
                and t.get("text-anchor") == "start"
            ]

        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        spaced = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(row_padding=20.0)))
            .to_svg()
        )
        base_ys = legend_label_ys(base)
        spaced_ys = legend_label_ys(spaced)
        assert len(base_ys) >= 2 and len(spaced_ys) >= 2
        base_gap = base_ys[1] - base_ys[0]
        spaced_gap = spaced_ys[1] - spaced_ys[0]
        assert spaced_gap > base_gap + 5.0, (
            f"per-channel row_padding must widen entry spacing (base gap {base_gap}, "
            f"spaced gap {spaced_gap})"
        )

    def test_columns_changes_render(self, color_df: pl.DataFrame) -> None:
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        multi = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(columns=2)))
            .to_svg()
        )
        assert multi != base, "per-channel columns=2 must change legend layout"

    def test_orient_bottom_changes_render(self, color_df: pl.DataFrame) -> None:
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        bottom = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(orient="bottom")))
            .to_svg()
        )
        assert bottom != base, "per-channel orient='bottom' must move the legend"

    def test_label_font_size_changes_render(self, color_df: pl.DataFrame) -> None:
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        sized = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend=fm.Legend(label_font_size=24.0)))
            .to_svg()
        )
        assert sized != base, "per-channel legend label_font_size must change the SVG"

    def test_title_font_size(self, color_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(color_df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                color=fm.Color("g:N", legend=fm.Legend(title="Cat", title_font_size=20.0)),
            )
            .to_svg()
        )
        titled = [
            t
            for t in _texts(_svg_root(svg))
            if (t.text or "").strip() == "Cat" and t.get("font-size") == "20"
        ]
        assert titled, "per-channel legend title_font_size must size the legend title"

    def test_gradient_length_changes_render(self) -> None:
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "v": [10.0, 50.0, 90.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="v:Q").to_svg()
        longer = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("v:Q", legend=fm.Legend(gradient_length=200.0)))
            .to_svg()
        )
        assert longer != base, "per-channel gradient_length must change a gradient legend"

    def test_tick_min_step_changes_render(self) -> None:
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "v": [10.0, 50.0, 90.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="v:Q").to_svg()
        stepped = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("v:Q", legend=fm.Legend(tick_min_step=40.0)))
            .to_svg()
        )
        assert stepped != base, "per-channel legend tick_min_step must change a gradient legend"


# ---------------------------------------------------------------------------
# Parity: per-channel and chart-level paths converge on the same SVG
#
# B5 root cause was that the per-channel path and the chart-level configure_*
# path "run in parallel and never meet" (RCA §3). These tests prove they now
# converge: setting a field per-channel on BOTH axes (or on the legend) yields
# the same observable SVG attribute as the chart-level configure_* call.
# ---------------------------------------------------------------------------


class TestPerChannelChartLevelParity:
    """Per-channel and chart-level styling produce identical rendered output."""

    def test_grid_color_parity(self, scatter_df: pl.DataFrame) -> None:
        per_channel = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(grid_color="#abc123")),
                y=fm.Y("y", axis=fm.Axis(grid_color="#abc123")),
            )
            .to_svg()
        )
        chart_level = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(grid_color="#abc123")
            .to_svg()
        )
        assert per_channel == chart_level, (
            "per-channel grid_color on both axes must equal configure_axis(grid_color=...)"
        )

    def test_domain_width_parity(self, scatter_df: pl.DataFrame) -> None:
        per_channel = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(domain_width=4.0)),
                y=fm.Y("y", axis=fm.Axis(domain_width=4.0)),
            )
            .to_svg()
        )
        chart_level = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(domain_width=4.0)
            .to_svg()
        )
        assert per_channel == chart_level, (
            "per-channel domain_width on both axes must equal configure_axis(domain_width=...)"
        )

    def test_label_angle_parity(self, scatter_df: pl.DataFrame) -> None:
        per_channel = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(label_angle=-30)),
                y=fm.Y("y", axis=fm.Axis(label_angle=-30)),
            )
            .to_svg()
        )
        chart_level = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(label_angle=-30)
            .to_svg()
        )
        assert per_channel == chart_level, (
            "per-channel label_angle on both axes must equal configure_axis(label_angle=...)"
        )

    def test_grid_opacity_parity(self, scatter_df: pl.DataFrame) -> None:
        per_channel = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(grid_opacity=0.3)),
                y=fm.Y("y", axis=fm.Axis(grid_opacity=0.3)),
            )
            .to_svg()
        )
        chart_level = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(grid_opacity=0.3)
            .to_svg()
        )
        assert per_channel == chart_level, (
            "per-channel grid_opacity on both axes must equal configure_axis(grid_opacity=...)"
        )


# ---------------------------------------------------------------------------
# Fail-loud: bad per-channel keys raise; camelCase aliases work
#
# The typed-spec per-channel path (B5 unit 1) deserializes into a serde struct
# that rejects unknown fields, so a typo surfaces as a ValueError at render
# rather than silently dropping (the original B5 failure mode).
# ---------------------------------------------------------------------------


class TestPerChannelFailLoud:
    """Misspelled per-channel keys raise; documented aliases keep working."""

    def test_misspelled_axis_key_raises(self, scatter_df: pl.DataFrame) -> None:
        chart = (
            fm.Chart(scatter_df).mark_point().encode(x=fm.X("x", axis={"grid_colr": "#f00"}), y="y")
        )
        with pytest.raises(ValueError, match="grid_colr"):
            chart.to_svg()

    def test_misspelled_legend_key_raises(self, color_df: pl.DataFrame) -> None:
        chart = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g:N", legend={"symbol_typ": "square"}))
        )
        with pytest.raises(ValueError, match="symbol_typ"):
            chart.to_svg()

    def test_camel_case_label_angle_alias(self, scatter_df: pl.DataFrame) -> None:
        """A camelCase ``labelAngle`` key is accepted via serde alias and rotates labels."""
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis={"labelAngle": -30}), y="y")
            .to_svg()
        )
        rotated = [t for t in _texts(_svg_root(svg)) if "rotate(-30" in (t.get("transform") or "")]
        assert rotated, "camelCase labelAngle alias must rotate x tick labels by -30"


# ---------------------------------------------------------------------------
# Golden stability: per-channel orphan fields must not perturb the no-style render
# ---------------------------------------------------------------------------


class TestGoldenStability:
    """Charts with no per-channel axis/legend styling render deterministically."""

    def test_no_styling_render_byte_identical(self, color_df: pl.DataFrame) -> None:
        """The default (no per-channel styling) render is byte-stable across calls."""
        chart = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N")
        baseline = chart.to_svg()
        assert chart.to_svg() == baseline

    def test_empty_axis_legend_no_op(self, color_df: pl.DataFrame) -> None:
        """``fm.Axis()`` / ``fm.Legend()`` with all defaults must not change the SVG.

        The new orphan fields default to ``None``; a bare value class must serialize
        empty and leave the render byte-identical to the no-style baseline.
        """
        baseline = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N").to_svg()
        styled = (
            fm.Chart(color_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis()),
                y=fm.Y("y", axis=fm.Axis()),
                color=fm.Color("g:N", legend=fm.Legend()),
            )
            .to_svg()
        )
        assert styled == baseline, (
            "bare fm.Axis()/fm.Legend() (all defaults) must not perturb the render"
        )
