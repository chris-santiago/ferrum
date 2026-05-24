"""Tests for ferrum.annotation — coords, primitives, and the Annotate container."""

from __future__ import annotations

import pytest

from ferrum.annotation.container import Annotate
from ferrum.annotation.coords import NormCoord, PixelCoord, norm, px
from ferrum.annotation.primitives import (
    AnnotationArrow,
    AnnotationBracket,
    AnnotationCallout,
    AnnotationImage,
    AnnotationLine,
    AnnotationRect,
    AnnotationSpan,
    AnnotationText,
    arrow,
    bracket,
    callout,
    image,
    line,
    rect,
    span,
    text,
)


# ---------------------------------------------------------------------------
# Coordinate wrappers
# ---------------------------------------------------------------------------


class TestPixelCoord:
    def test_px_factory(self):
        coord = px(50)
        assert coord == PixelCoord(value=50)

    def test_px_float(self):
        coord = px(3.14)
        assert coord.value == 3.14

    def test_frozen(self):
        coord = px(10)
        with pytest.raises((TypeError, AttributeError)):
            coord.value = 99  # type: ignore[misc]


class TestNormCoord:
    def test_norm_factory(self):
        coord = norm(0.5)
        assert coord == NormCoord(value=0.5)

    def test_norm_zero(self):
        coord = norm(0.0)
        assert coord.value == 0.0

    def test_norm_one(self):
        coord = norm(1.0)
        assert coord.value == 1.0

    def test_frozen(self):
        coord = norm(0.5)
        with pytest.raises((TypeError, AttributeError)):
            coord.value = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coord serialization helper (tested via to_dict)
# ---------------------------------------------------------------------------


class TestCoordSerialization:
    """Verify _coord() serialization through primitive to_dict()."""

    def test_plain_float_stays_float(self):
        t = text(1.5, 2.5, "hello")
        d = t.to_dict()
        assert d["x"] == 1.5
        assert d["y"] == 2.5

    def test_pixel_coord_serialized_as_dict(self):
        t = text(px(50), px(100), "hello")
        d = t.to_dict()
        assert d["x"] == {"px": 50}
        assert d["y"] == {"px": 100}

    def test_norm_coord_serialized_as_dict(self):
        t = text(norm(0.25), norm(0.75), "hello")
        d = t.to_dict()
        assert d["x"] == {"norm": 0.25}
        assert d["y"] == {"norm": 0.75}


# ---------------------------------------------------------------------------
# AnnotationText
# ---------------------------------------------------------------------------


class TestAnnotationText:
    def test_defaults(self):
        t = text(1.0, 2.0, "label")
        assert isinstance(t, AnnotationText)
        assert t.x == 1.0
        assert t.y == 2.0
        assert t.text == "label"
        assert t.font_size == 12
        assert t.color == "#333"
        assert t.anchor == "start"
        assert t.baseline == "middle"
        assert t.angle == 0
        assert t.dx == 0
        assert t.dy == 0
        assert t.z == "above_marks"

    def test_frozen(self):
        t = text(0, 0, "x")
        with pytest.raises((TypeError, AttributeError)):
            t.color = "red"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = text(0, 0, "hello").to_dict()
        assert d["type"] == "text"

    def test_to_dict_custom_fields(self):
        t = text(1, 2, "hi", font_size=14, color="blue", angle=45, dx=5, dy=-3)
        d = t.to_dict()
        assert d["font_size"] == 14
        assert d["color"] == "blue"
        assert d["angle"] == 45
        assert d["dx"] == 5
        assert d["dy"] == -3

    def test_to_dict_coord_serialization(self):
        t = text(px(10), norm(0.5), "mixed")
        d = t.to_dict()
        assert d["x"] == {"px": 10}
        assert d["y"] == {"norm": 0.5}


# ---------------------------------------------------------------------------
# AnnotationArrow
# ---------------------------------------------------------------------------


class TestAnnotationArrow:
    def test_defaults(self):
        a = arrow(0, 0, 1, 1)
        assert isinstance(a, AnnotationArrow)
        assert a.stroke == "#333"
        assert a.stroke_width == 1.5
        assert a.head_size == 8
        assert a.curve == "straight"

    def test_frozen(self):
        a = arrow(0, 0, 1, 1)
        with pytest.raises((TypeError, AttributeError)):
            a.stroke = "blue"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = arrow(0, 0, 1, 1).to_dict()
        assert d["type"] == "arrow"

    def test_to_dict_endpoints(self):
        a = arrow(px(0), 1.0, norm(0.5), px(20))
        d = a.to_dict()
        assert d["x"] == {"px": 0}
        assert d["y"] == 1.0
        assert d["x2"] == {"norm": 0.5}
        assert d["y2"] == {"px": 20}

    def test_to_dict_custom_fields(self):
        a = arrow(0, 0, 1, 1, stroke="red", stroke_width=2, head_size=12, curve="arc")
        d = a.to_dict()
        assert d["stroke"] == "red"
        assert d["stroke_width"] == 2
        assert d["head_size"] == 12
        assert d["curve"] == "arc"


# ---------------------------------------------------------------------------
# AnnotationRect
# ---------------------------------------------------------------------------


class TestAnnotationRect:
    def test_defaults(self):
        r = rect(0, 0, 1, 1, fill="#eee")
        assert isinstance(r, AnnotationRect)
        assert r.fill == "#eee"
        assert r.opacity == 0.1
        assert r.stroke is None
        assert r.corner_radius == 0

    def test_frozen(self):
        r = rect(0, 0, 1, 1, fill="#eee")
        with pytest.raises((TypeError, AttributeError)):
            r.fill = "blue"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = rect(0, 0, 1, 1, fill="#eee").to_dict()
        assert d["type"] == "rect"

    def test_to_dict_omits_stroke_when_none(self):
        d = rect(0, 0, 1, 1, fill="#eee").to_dict()
        assert "stroke" not in d

    def test_to_dict_includes_stroke_when_set(self):
        d = rect(0, 0, 1, 1, fill="#eee", stroke="#333").to_dict()
        assert d["stroke"] == "#333"

    def test_to_dict_coord_serialization(self):
        r = rect(px(10), norm(0.1), px(50), norm(0.9), fill="red")
        d = r.to_dict()
        assert d["x1"] == {"px": 10}
        assert d["y1"] == {"norm": 0.1}
        assert d["x2"] == {"px": 50}
        assert d["y2"] == {"norm": 0.9}


# ---------------------------------------------------------------------------
# AnnotationLine
# ---------------------------------------------------------------------------


class TestAnnotationLine:
    def test_defaults(self):
        ln = line(0, 0, 1, 1)
        assert isinstance(ln, AnnotationLine)
        assert ln.stroke == "#333"
        assert ln.stroke_width == 1
        assert ln.dash is None

    def test_frozen(self):
        ln = line(0, 0, 1, 1)
        with pytest.raises((TypeError, AttributeError)):
            ln.stroke = "green"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = line(0, 0, 1, 1).to_dict()
        assert d["type"] == "line"

    def test_to_dict_omits_dash_when_none(self):
        d = line(0, 0, 1, 1).to_dict()
        assert "dash" not in d

    def test_to_dict_includes_dash_when_set(self):
        d = line(0, 0, 1, 1, dash=[4, 4]).to_dict()
        assert d["dash"] == [4, 4]

    def test_to_dict_coord_serialization(self):
        ln = line(px(5), 0.0, norm(1.0), px(20))
        d = ln.to_dict()
        assert d["x1"] == {"px": 5}
        assert d["y1"] == 0.0
        assert d["x2"] == {"norm": 1.0}
        assert d["y2"] == {"px": 20}


# ---------------------------------------------------------------------------
# AnnotationSpan
# ---------------------------------------------------------------------------


class TestAnnotationSpan:
    def test_defaults(self):
        s = span("x", 0, 1, fill="#eee")
        assert isinstance(s, AnnotationSpan)
        assert s.axis == "x"
        assert s.start == 0
        assert s.end == 1
        assert s.fill == "#eee"
        assert s.opacity == 0.3
        assert s.label is None
        assert s.label_position == "top"

    def test_frozen(self):
        s = span("y", 0, 1, fill="#eee")
        with pytest.raises((TypeError, AttributeError)):
            s.fill = "blue"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = span("x", 0, 1, fill="#eee").to_dict()
        assert d["type"] == "span"

    def test_to_dict_omits_label_when_none(self):
        d = span("x", 0, 1, fill="#eee").to_dict()
        assert "label" not in d

    def test_to_dict_includes_label_when_set(self):
        d = span("y", 5, 10, fill="#ccc", label="region").to_dict()
        assert d["label"] == "region"

    def test_to_dict_coord_serialization(self):
        s = span("x", px(10), px(50), fill="blue")
        d = s.to_dict()
        assert d["start"] == {"px": 10}
        assert d["end"] == {"px": 50}


# ---------------------------------------------------------------------------
# AnnotationBracket
# ---------------------------------------------------------------------------


class TestAnnotationBracket:
    def test_defaults(self):
        b = bracket(0, 0, 1, 0, label="group")
        assert isinstance(b, AnnotationBracket)
        assert b.label == "group"
        assert b.direction == "above"
        assert b.stroke == "#333"
        assert b.tip_length == 6

    def test_frozen(self):
        b = bracket(0, 0, 1, 0, label="g")
        with pytest.raises((TypeError, AttributeError)):
            b.label = "other"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = bracket(0, 0, 1, 0, label="g").to_dict()
        assert d["type"] == "bracket"

    def test_to_dict_all_fields(self):
        b = bracket(0, 0, 1, 0, label="A–B", direction="below", stroke="navy", tip_length=8)
        d = b.to_dict()
        assert d["label"] == "A–B"
        assert d["direction"] == "below"
        assert d["stroke"] == "navy"
        assert d["tip_length"] == 8

    def test_to_dict_coord_serialization(self):
        b = bracket(px(0), norm(0.0), px(100), norm(0.0), label="span")
        d = b.to_dict()
        assert d["x1"] == {"px": 0}
        assert d["y1"] == {"norm": 0.0}
        assert d["x2"] == {"px": 100}
        assert d["y2"] == {"norm": 0.0}


# ---------------------------------------------------------------------------
# AnnotationCallout
# ---------------------------------------------------------------------------


class TestAnnotationCallout:
    def test_defaults(self):
        c = callout(1.0, 2.0, "peak")
        assert isinstance(c, AnnotationCallout)
        assert c.x == 1.0
        assert c.y == 2.0
        assert c.text == "peak"
        assert c.text_x is None
        assert c.text_y is None
        assert c.arrow == "curved"
        assert c.padding == 4
        assert c.background == "#fff"
        assert c.border_color == "#ccc"
        assert c.border_radius == 3

    def test_frozen(self):
        c = callout(0, 0, "note")
        with pytest.raises((TypeError, AttributeError)):
            c.text = "other"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = callout(0, 0, "note").to_dict()
        assert d["type"] == "callout"

    def test_to_dict_omits_text_xy_when_none(self):
        d = callout(0, 0, "note").to_dict()
        assert "text_x" not in d
        assert "text_y" not in d

    def test_to_dict_includes_text_xy_when_set(self):
        c = callout(0, 0, "note", text_x=px(50), text_y=norm(0.8))
        d = c.to_dict()
        assert d["text_x"] == {"px": 50}
        assert d["text_y"] == {"norm": 0.8}

    def test_to_dict_coord_serialization(self):
        c = callout(px(30), norm(0.4), "here")
        d = c.to_dict()
        assert d["x"] == {"px": 30}
        assert d["y"] == {"norm": 0.4}


# ---------------------------------------------------------------------------
# AnnotationImage
# ---------------------------------------------------------------------------


class TestAnnotationImage:
    def test_defaults(self):
        img = image(1.0, 2.0, "https://example.com/logo.png")
        assert isinstance(img, AnnotationImage)
        assert img.src == "https://example.com/logo.png"
        assert img.width == 50
        assert img.height == 50
        assert img.anchor == "center"

    def test_frozen(self):
        img = image(0, 0, "img.png")
        with pytest.raises((TypeError, AttributeError)):
            img.src = "other.png"  # type: ignore[misc]

    def test_to_dict_type_key(self):
        d = image(0, 0, "img.png").to_dict()
        assert d["type"] == "image"

    def test_to_dict_all_fields(self):
        img = image(px(10), norm(0.5), "data:base64,abc", width=80, height=60, anchor="top-left")
        d = img.to_dict()
        assert d["x"] == {"px": 10}
        assert d["y"] == {"norm": 0.5}
        assert d["src"] == "data:base64,abc"
        assert d["width"] == 80
        assert d["height"] == 60
        assert d["anchor"] == "top-left"


# ---------------------------------------------------------------------------
# Annotate container
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_single_item_wrapped_in_list(self):
        t = text(0, 0, "hi")
        ann = Annotate(t)
        assert ann.items == [t]

    def test_list_of_items_stored(self):
        t1 = text(0, 0, "a")
        t2 = text(1, 1, "b")
        ann = Annotate([t1, t2])
        assert ann.items == [t1, t2]

    def test_list_is_copied(self):
        t1 = text(0, 0, "a")
        original = [t1]
        ann = Annotate(original)
        original.append(text(1, 1, "b"))
        # The Annotate should still hold only the original item
        assert len(ann.items) == 1

    def test_to_dict_list(self):
        t = text(1.0, 2.0, "label")
        s = span("x", 0, 5, fill="#eee")
        ann = Annotate([t, s])
        result = ann.to_dict_list()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "span"

    def test_to_dict_list_single_item(self):
        a = arrow(0, 0, 1, 1)
        ann = Annotate(a)
        result = ann.to_dict_list()
        assert len(result) == 1
        assert result[0]["type"] == "arrow"

    def test_frozen(self):
        ann = Annotate(text(0, 0, "x"))
        with pytest.raises((TypeError, AttributeError)):
            ann.items = []  # type: ignore[misc]

    def test_heterogeneous_primitives(self):
        items = [
            text(0, 0, "t"),
            arrow(0, 0, 1, 1),
            rect(0, 0, 1, 1, fill="red"),
            line(0, 0, 1, 1),
            span("x", 0, 1, fill="blue"),
            bracket(0, 0, 1, 0, label="g"),
            callout(0, 0, "note"),
            image(0, 0, "img.png"),
        ]
        ann = Annotate(items)
        result = ann.to_dict_list()
        types = [d["type"] for d in result]
        assert types == ["text", "arrow", "rect", "line", "span", "bracket", "callout", "image"]
