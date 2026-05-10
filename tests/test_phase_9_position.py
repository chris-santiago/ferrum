"""Phase 9c position-adjustment tests."""
from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fe
from ferrum import Dodge, Identity


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
