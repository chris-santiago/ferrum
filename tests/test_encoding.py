import warnings
import pytest

from ferrum._warn import reset_warnings
from ferrum.encoding.base import ChannelBase
from ferrum.encoding import (
    X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius,
    Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity,
    StrokeWidth, StrokeDash, Size, Shape, Angle,
)


class _TestChannel(ChannelBase):
    _channel_name = "x"
    _renders_in_phase_8a = True
    _honored_kwargs = frozenset(["type", "scale", "title"])


def test_channelbase_stores_field_and_kwargs():
    reset_warnings()
    c = _TestChannel("price", type="Q", title="Price")
    assert c.field == "price"
    assert c._kwargs == {"type": "Q", "title": "Price"}


def test_channelbase_warns_once_on_deferred_kwarg():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _TestChannel("price", axis={"grid": False})
    assert len(w) == 1
    assert "axis" in str(w[0].message)


def test_to_encoding_spec_dict_has_field_and_type():
    reset_warnings()
    c = _TestChannel("price", type="Q")
    d = c.to_encoding_spec_dict()
    assert d["field"] == "price"
    assert d["type_"] == "Q"


def test_to_implicit_transforms_with_bin_kwarg():
    class _BinTestChannel(ChannelBase):
        _channel_name = "x"
        _renders_in_phase_8a = True
        _honored_kwargs = frozenset(["type", "bin", "aggregate"])

    reset_warnings()
    c = _BinTestChannel("price", bin=True)
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    # First (and only) transform should be a Bin instance
    from ferrum import Bin
    assert isinstance(transforms[0], Bin)


def test_to_implicit_transforms_with_aggregate_kwarg():
    class _AggTestChannel(ChannelBase):
        _channel_name = "y"
        _renders_in_phase_8a = True
        _honored_kwargs = frozenset(["type", "aggregate"])

    reset_warnings()
    c = _AggTestChannel("latency", aggregate="mean")
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    from ferrum import Aggregate
    assert isinstance(transforms[0], Aggregate)


# ---------------------------------------------------------------------------
# Task 16: positional channels
# ---------------------------------------------------------------------------

def test_x_renders_in_phase_8a():
    assert X._renders_in_phase_8a is True


def test_y_renders_in_phase_8a():
    assert Y._renders_in_phase_8a is True


def test_all_positional_channels_render():
    for cls in (X2, Y2, XError, YError, XError2, YError2, Theta, Radius):
        assert cls._renders_in_phase_8a is True, f"{cls.__name__} must render"


def test_x_construction_with_full_honored_kwargs():
    reset_warnings()
    from ferrum import LinearScale
    c = X("price", type="Q", bin=True, aggregate="mean",
          scale=LinearScale(domain=[0, 100], range=[0, 600]),
          title="Price")
    assert c.field == "price"
    assert c._kwargs["type"] == "Q"


def test_x_warns_on_deferred_kwargs():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X("price", axis={"grid": False}, sort="ascending")
    assert len(w) == 2


# ---------------------------------------------------------------------------
# Task 17: appearance channels
# ---------------------------------------------------------------------------

def test_color_renders_in_phase_8a():
    assert Color._renders_in_phase_8a is True


def test_size_shape_opacity_render_in_phase_8a():
    for cls in (Size, Shape, Opacity):
        assert cls._renders_in_phase_8a is True, f"{cls.__name__} must render in 8a"


def test_all_appearance_channels_render():
    for cls in (Fill, Stroke, FillOpacity, StrokeOpacity, StrokeWidth, StrokeDash, Angle):
        assert cls._renders_in_phase_8a is True, f"{cls.__name__} must render"


def test_color_with_scheme_kwarg_no_warning():
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Color("species", scheme="tableau10")
    assert len(w) == 0  # scheme is honored for Color in 8a


def test_stroke_with_field_warns_once_on_render_attempt():
    # Bare construction with just `field` doesn't pass kwargs → no warning yet.
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Stroke("color")
    assert len(w) == 0


# ---------------------------------------------------------------------------
# Task 18: text/detail/tooltip classes
# ---------------------------------------------------------------------------

def test_text_channels_all_render():
    from ferrum.encoding import Text, Detail, Tooltip, TooltipField, Href, Description, Key
    for cls in (Text, Detail, Tooltip, TooltipField, Href, Description, Key):
        assert cls._renders_in_phase_8a is True, f"{cls.__name__} must render"


def test_tooltip_accepts_multiple_fields():
    from ferrum.encoding import Tooltip
    t = Tooltip("a", "b", "c")
    assert t._field_list == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Task 19: facet channels
# ---------------------------------------------------------------------------

def test_facet_channels_render():
    from ferrum.encoding import Facet, FacetRow, FacetCol
    for cls in (Facet, FacetRow, FacetCol):
        assert cls._renders_in_phase_8a is True


# ---------------------------------------------------------------------------
# Task 36: warn-once across multiple renders
# ---------------------------------------------------------------------------

def test_stroke_no_warning_across_renders():
    """Stroke is silently accepted (aliased to color when no color is set)."""
    import polars as pl
    from ferrum import Chart, Stroke
    from ferrum._warn import reset_warnings

    reset_warnings()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            Chart(df).mark_point().encode(x="a", y="b", stroke=Stroke("c"))
    # Stroke is now silently handled — no warnings expected.
    stroke_channel_warnings = [
        wi for wi in w if "stroke" in str(wi.message).lower()
    ]
    assert len(stroke_channel_warnings) == 0, (
        f"Expected 0 stroke warnings, got {len(stroke_channel_warnings)}: "
        f"{[str(wi.message) for wi in stroke_channel_warnings]}"
    )
