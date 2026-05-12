"""Phase 9c position-adjustment tests."""
from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fe
from ferrum import Dodge, Identity, Jitter, Stack


# ---------------------------------------------------------------------------
# Task 20 — Identity + Dodge
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_to_spec_dict(self):
        assert Identity().to_spec_dict() == {"type": "identity"}

    def test_immutable(self):
        with pytest.raises(AttributeError):
            Identity().foo = 1  # type: ignore[attr-defined]

    def test_equality_and_hash(self):
        assert Identity() == Identity()
        assert hash(Identity()) == hash(Identity())


class TestDodge:
    def test_to_spec_dict_with_by(self):
        d = Dodge(by="species", padding=0.1)
        assert d.to_spec_dict() == {
            "type": "dodge",
            "padding": 0.1,
            "by": "species",
        }

    def test_to_spec_dict_no_by(self):
        d = Dodge()
        assert d.to_spec_dict() == {"type": "dodge", "padding": 0.05}

    def test_immutable(self):
        d = Dodge(by="g")
        with pytest.raises(AttributeError):
            d.by = "h"

    def test_invalid_padding_errors(self):
        with pytest.raises(ValueError, match="padding"):
            Dodge(padding=1.5)
        with pytest.raises(ValueError, match="padding"):
            Dodge(padding=-0.1)


class TestPositionEligibility:
    def test_identity_accepted_by_all_marks(self):
        df = pl.DataFrame({"x": ["a", "b"], "y": [3.0, 4.0]})
        for mark_name in ("bar", "point", "rule", "line", "tick", "rect"):
            method = getattr(fe.Chart(df), f"mark_{mark_name}")
            method(position=Identity()).encode(x="x", y="y")

    def test_dodge_rejected_by_line(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "g": ["a", "b"]})
        with pytest.raises(TypeError, match="Dodge"):
            fe.Chart(df).mark_line(position=Dodge(by="g"))

    def test_dodge_accepted_by_bar(self):
        df = pl.DataFrame({"x": ["a", "b"], "y": [3.0, 4.0], "g": ["a", "b"]})
        chart = fe.Chart(df).mark_bar(position=Dodge(by="g")).encode(
            x="x", y="y", color="g"
        )
        spec = chart.to_spec()
        d = json.loads(spec.to_json())
        assert d.get("position", {}).get("type") == "dodge"
        assert d.get("position", {}).get("by") == "g"


@pytest.mark.parametrize(
    "hue_field,categories",
    [("g", ["a", "b"])],
)
def test_dodge_renders_side_by_side(hue_field, categories):
    """``mark_bar(position=Dodge)`` must produce one rect per (cat, group)."""
    rows = []
    for cat in ("X", "Y", "Z"):
        for g in categories:
            rows.append({"cat": cat, "g": g, "v": float(ord(cat) + len(g))})
    df = pl.DataFrame(rows)
    chart = fe.Chart(df).mark_bar(position=Dodge(by=hue_field)).encode(
        x="cat", y="v", color=hue_field
    )
    svg = chart.show_svg()
    assert "<svg" in svg
    # 3 categories × 2 groups = 6 rows → 6 bars.
    assert svg.count("<rect") >= 6


# ---------------------------------------------------------------------------
# Task 21 — Jitter
# ---------------------------------------------------------------------------

class TestJitter:
    def test_construction(self):
        j = Jitter(axis="x", width=0.5, seed=42)
        assert j.axis == "x"
        assert j.width == 0.5
        assert j.seed == 42

    def test_default_axis_x_width_0_4(self):
        j = Jitter()
        assert j.axis == "x"
        assert j.width == 0.4
        assert j.seed is None

    def test_to_spec_dict_with_seed(self):
        j = Jitter(axis="both", width=0.3, seed=7)
        assert j.to_spec_dict() == {
            "type": "jitter",
            "axis": "both",
            "width": 0.3,
            "seed": 7,
        }

    def test_to_spec_dict_no_seed(self):
        j = Jitter()
        assert j.to_spec_dict() == {"type": "jitter", "axis": "x", "width": 0.4}

    def test_invalid_axis_errors(self):
        with pytest.raises(ValueError, match="axis"):
            Jitter(axis="invalid")

    def test_invalid_width_errors(self):
        with pytest.raises(ValueError, match="width"):
            Jitter(width=0.0)

    def test_eligibility_rejects_bar(self):
        df = pl.DataFrame({"x": [1, 2]})
        with pytest.raises(TypeError, match="Jitter"):
            fe.Chart(df).mark_bar(position=Jitter())

    def test_eligibility_accepts_point(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        fe.Chart(df).mark_point(position=Jitter()).encode(x="x", y="y")

    def test_renders_with_explicit_seed_byte_identical(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        c = (
            fe.Chart(df)
            .mark_point(position=Jitter(width=0.3, seed=42))
            .encode(x="x", y="y")
        )
        a = c.show_svg()
        b = c.show_svg()
        assert a == b  # identical seed → identical output

    def test_renders_with_seed_none_byte_identical(self):
        # seed=None falls back to xxh3 hash of (x, y) per row — also byte-deterministic.
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        c = (
            fe.Chart(df)
            .mark_point(position=Jitter(width=0.3))
            .encode(x="x", y="y")
        )
        a = c.show_svg()
        b = c.show_svg()
        assert a == b

    def test_different_seeds_produce_different_output(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        a = (
            fe.Chart(df)
            .mark_point(position=Jitter(width=0.5, seed=1))
            .encode(x="x", y="y")
            .show_svg()
        )
        b = (
            fe.Chart(df)
            .mark_point(position=Jitter(width=0.5, seed=2))
            .encode(x="x", y="y")
            .show_svg()
        )
        assert a != b


# ---------------------------------------------------------------------------
# Task 22 — Stack
# ---------------------------------------------------------------------------

class TestStack:
    def test_construction_default(self):
        s = Stack()
        assert s.by is None
        assert s.offset == "zero"

    def test_to_spec_dict_normalize(self):
        s = Stack(by="hue", offset="normalize")
        assert s.to_spec_dict() == {
            "type": "stack",
            "offset": "normalize",
            "anchor": "top",
            "by": "hue",
        }

    def test_to_spec_dict_anchor_mid(self):
        # Schwabish C6 audit-rework (2026-05-12): anchor="mid" routes
        # annotation marks (text/point/rule/tick) to segment midpoints.
        s = Stack(by="actual", offset="zero", anchor="mid")
        assert s.to_spec_dict() == {
            "type": "stack",
            "offset": "zero",
            "anchor": "mid",
            "by": "actual",
        }

    def test_invalid_anchor_errors(self):
        with pytest.raises(ValueError, match="anchor"):
            Stack(anchor="diagonal")

    def test_invalid_offset_errors(self):
        with pytest.raises(ValueError, match="offset"):
            Stack(offset="bogus")

    def test_eligibility_accepts_annotation_marks(self):
        # Schwabish SB-followup (2026-05-12): Stack now accepts the
        # annotation-style marks (text, point, rule, tick); their y
        # values map to per-segment midpoints so annotations land at
        # the visual centre of each stacked-bar segment.
        df = pl.DataFrame({"x": ["a", "b"], "y": [3.0, 4.0]})
        fe.Chart(df).mark_text(position=Stack()).encode(x="x", y="y")
        fe.Chart(df).mark_point(position=Stack()).encode(x="x", y="y")
        fe.Chart(df).mark_rule(position=Stack()).encode(x="x", y="y")
        fe.Chart(df).mark_tick(position=Stack()).encode(x="x", y="y")

    def test_eligibility_accepts_bar_area_ribbon(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        fe.Chart(df).mark_bar(position=Stack()).encode(x="x", y="y")
        fe.Chart(df).mark_area(position=Stack()).encode(x="x", y="y")

    def test_stack_zero_renders(self):
        df = pl.DataFrame({
            "x": ["1", "2", "1", "2"],
            "y": [10.0, 20.0, 5.0, 8.0],
            "g": ["a", "a", "b", "b"],
        })
        chart = (
            fe.Chart(df)
            .mark_bar(position=Stack(by="g"))
            .encode(x="x", y="y", color="g")
        )
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_stack_normalize_total_y_is_one(self):
        df = pl.DataFrame({
            "x": ["1", "1", "2", "2"],
            "y": [3.0, 7.0, 1.0, 9.0],
            "g": ["a", "b", "a", "b"],
        })
        chart = (
            fe.Chart(df)
            .mark_bar(position=Stack(by="g", offset="normalize"))
            .encode(x="x", y="y", color="g")
        )
        svg = chart.show_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Task 23 — eligibility-matrix enforcement across every Chart.mark_* method
# ---------------------------------------------------------------------------

class TestEligibilityMatrix:
    @pytest.mark.parametrize(
        "mark_name,position_class,should_accept",
        [
            ("bar",       "Identity", True),
            ("bar",       "Dodge",    True),
            ("bar",       "Jitter",   False),
            ("bar",       "Stack",    True),
            ("point",     "Dodge",    True),
            ("point",     "Jitter",   True),
            # Schwabish SB-followup (2026-05-12): point/text/rule/tick
            # accept Stack — y maps to segment midpoint for
            # annotations on stacked bars.
            ("point",     "Stack",    True),
            ("line",      "Dodge",    False),
            ("line",      "Jitter",   False),
            ("line",      "Stack",    False),
            ("rule",      "Dodge",    False),
            ("rule",      "Jitter",   False),
            ("rule",      "Stack",    True),
            ("area",      "Stack",    True),
            ("area",      "Dodge",    False),
            ("ribbon",    "Stack",    True),
            ("ribbon",    "Dodge",    True),
            ("tick",      "Jitter",   True),
            ("tick",      "Dodge",    False),
            ("tick",      "Stack",    True),
            ("swarm",     "Jitter",   True),
            ("swarm",     "Dodge",    True),
            ("violin",    "Dodge",    True),
            ("violin",    "Jitter",   False),
            ("boxplot",   "Dodge",    True),
            ("boxplot",   "Stack",    False),
            ("errorbar",  "Dodge",    True),
            ("errorbar",  "Jitter",   False),
            ("errorband", "Dodge",    True),
            ("errorband", "Stack",    False),
        ],
    )
    def test_eligibility(self, mark_name, position_class, should_accept):
        position_classes = {
            "Identity": Identity(),
            "Dodge": Dodge(),
            "Jitter": Jitter(),
            "Stack": Stack(),
        }
        pos = position_classes[position_class]
        df = pl.DataFrame({
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "y2": [5.0, 6.0],
            "g": ["a", "b"],
        })
        method = getattr(fe.Chart(df), f"mark_{mark_name}")
        if should_accept:
            method(position=pos)
        else:
            with pytest.raises(TypeError, match=position_class):
                method(position=pos)
