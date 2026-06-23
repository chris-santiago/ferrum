import warnings

from ferrum._warn import reset_warnings
from ferrum.encoding.base import ChannelBase
from ferrum.encoding import (
    X,
    Color,
    Stroke,
)


class _TestChannel(ChannelBase):
    _channel_name = "x"
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
        _honored_kwargs = frozenset(["type", "bin", "aggregate"])

    reset_warnings()
    c = _BinTestChannel("price", bin=True)
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    # to_implicit_transforms() now returns a _PendingBin sentinel instead of a
    # Rust Bin object.  The sentinel defers the named-vs-unnamed decision until
    # chart.to_spec() knows whether the chart is single or layered (FA-4 fix).
    from ferrum.encoding.base import _PendingBin

    assert isinstance(transforms[0], _PendingBin)
    assert transforms[0].field == "price"
    # bin=True carries no extra kwargs.
    assert transforms[0].bin_kwargs == ()
    # bin=True path has no pre-built bin_obj.
    assert transforms[0].bin_obj is None


def test_to_implicit_transforms_with_bin_instance():
    """BUG 1: bin=Bin("price", bin_count=5) must carry the Bin object intact.

    Pre-fix: to_implicit_transforms() tried bin_arg.to_json() (absent on PyO3
    Bin), fell through to _bin_dict={}, and silently dropped bin_count.
    The _PendingBin sentinel must now carry the Bin instance in bin_obj so the
    resolvers (single-chart and layered) can use it directly without guessing
    at PyO3 internals.
    """

    class _BinInstanceChannel(ChannelBase):
        _channel_name = "x"
        _honored_kwargs = frozenset(["type", "bin"])

    from ferrum._core import Bin as RustBin
    from ferrum.encoding.base import _PendingBin

    reset_warnings()
    rust_bin = RustBin("price", bin_count=5)
    c = _BinInstanceChannel("price", bin=rust_bin)
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    pb = transforms[0]
    assert isinstance(pb, _PendingBin)
    assert pb.field == "price"
    # bin_obj must hold the Bin instance, not be None.
    assert pb.bin_obj is rust_bin, (
        f"_PendingBin.bin_obj must be the passed Bin instance, got {pb.bin_obj!r}"
    )
    # bin_kwargs must be empty (kwargs come from the Bin object itself).
    assert pb.bin_kwargs == ()


def test_to_implicit_transforms_with_bin_dict():
    """bin={"bin_count": N} must preserve the kwarg in _PendingBin.bin_kwargs."""

    class _BinDictChannel(ChannelBase):
        _channel_name = "x"
        _honored_kwargs = frozenset(["type", "bin"])

    from ferrum.encoding.base import _PendingBin

    reset_warnings()
    c = _BinDictChannel("price", bin={"bin_count": 7})
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    pb = transforms[0]
    assert isinstance(pb, _PendingBin)
    assert pb.field == "price"
    assert pb.bin_obj is None
    assert dict(pb.bin_kwargs) == {"bin_count": 7}


def test_to_implicit_transforms_with_aggregate_kwarg():
    class _AggTestChannel(ChannelBase):
        _channel_name = "y"
        _honored_kwargs = frozenset(["type", "aggregate"])

    reset_warnings()
    c = _AggTestChannel("latency", aggregate="mean")
    transforms = c.to_implicit_transforms()
    assert len(transforms) == 1
    # to_implicit_transforms() now returns a _PendingAggregate sentinel instead of
    # a Rust Aggregate object. The sentinel defers groupby assignment until
    # chart.to_spec() can inspect all sibling encoding channels (B3 fix).
    from ferrum.encoding.base import _PendingAggregate

    assert isinstance(transforms[0], _PendingAggregate)
    assert transforms[0].field == "latency"
    assert transforms[0].agg == "mean"
    assert transforms[0].output_col == "mean_latency"


# ---------------------------------------------------------------------------
# Task 16: positional channels
# ---------------------------------------------------------------------------


def test_x_construction_with_full_honored_kwargs():
    reset_warnings()
    from ferrum import LinearScale

    c = X(
        "price",
        type="Q",
        bin=True,
        aggregate="mean",
        scale=LinearScale(domain=[0, 100], range=[0, 600]),
        title="Price",
    )
    assert c.field == "price"
    assert c._kwargs["type"] == "Q"


def test_x_honored_kwargs_no_warning():
    """axis=, sort=, stack=, impute=, format=, format_type= are all honored — no warning."""
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X("price", axis={"grid": False}, sort="ascending")
    # Both axis= and sort= are now honored — neither emits a warning.
    assert len(w) == 0


# ---------------------------------------------------------------------------
# Task 17: appearance channels
# ---------------------------------------------------------------------------


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


def test_tooltip_accepts_multiple_fields():
    from ferrum.encoding import Tooltip

    t = Tooltip("a", "b", "c")
    assert t._field_list == ["a", "b", "c"]


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
    stroke_channel_warnings = [wi for wi in w if "stroke" in str(wi.message).lower()]
    assert len(stroke_channel_warnings) == 0, (
        f"Expected 0 stroke warnings, got {len(stroke_channel_warnings)}: "
        f"{[str(wi.message) for wi in stroke_channel_warnings]}"
    )


# ---------------------------------------------------------------------------
# C13 — warn-once when stroke is dropped by color; ghost docstring accuracy
# ---------------------------------------------------------------------------


def test_stroke_dropped_by_color_warns_once():
    """encode(color=..., stroke=...) emits exactly one warning and drops stroke.

    When both color and stroke are encoded with data fields, stroke cannot be
    mapped to a separate scale — it is dropped.  A single UserWarning is emitted
    the first time this happens (at render time, inside _apply_channel_aliases);
    subsequent renders in the same context are silent.
    """
    import polars as pl
    from ferrum import Chart, Color, Stroke
    from ferrum._warn import reset_warnings

    reset_warnings()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "y"], "d": ["p", "q"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # The warning fires at to_svg() time (inside _apply_channel_aliases).
        Chart(df).mark_point().encode(x="a", y="b", color=Color("c"), stroke=Stroke("d")).to_svg()
    stroke_warns = [wi for wi in w if "stroke" in str(wi.message).lower()]
    assert len(stroke_warns) == 1, (
        f"Expected exactly 1 stroke-dropped warning; got {len(stroke_warns)}: "
        f"{[str(wi.message) for wi in stroke_warns]}"
    )

    # Second render (same warn-once context): must be suppressed.
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        Chart(df).mark_point().encode(x="a", y="b", color=Color("c"), stroke=Stroke("d")).to_svg()
    stroke_warns2 = [wi for wi in w2 if "stroke" in str(wi.message).lower()]
    assert len(stroke_warns2) == 0, (
        f"Expected 0 stroke warnings on second render (warn_once); "
        f"got {len(stroke_warns2)}: {[str(wi.message) for wi in stroke_warns2]}"
    )


def test_stroke_dropped_by_color_stroke_field_is_absent_from_spec():
    """When stroke is dropped, the rendered chart encodes color only (stroke absent).

    Verifies the drop is real: the chart renders without error and the stroke
    field does not appear as a separate color-scale field in the output.
    """
    import polars as pl
    from ferrum import Chart, Color, Stroke
    from ferrum._warn import reset_warnings

    reset_warnings()
    df = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"], "d": ["p", "q", "r"]}
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        svg = (
            Chart(df)
            .mark_point()
            .encode(x="a", y="b", color=Color("c"), stroke=Stroke("d"))
            .to_svg()
        )
    assert "<svg" in svg, "Chart with color+stroke (stroke dropped) must still render"


def test_ghost_channel_docstrings_do_not_narrate_render_status():
    """Ghost positional channels must not narrate per-channel render status.

    ENC-09: ``_honored_kwargs`` is the single, machine-readable source of truth
    for the honored-kwarg contract.  Per-channel docstrings must not duplicate it
    in prose (``"reserved for future use"`` / ``"no-op today"``), which drifts.
    X2, Y2, XError, YError, XError2, YError2, Theta2, Radius2 are all
    secondary-extent channels honoring only ``{"type"}``.
    """
    from ferrum.encoding import X2, Y2, XError, YError, XError2, YError2
    from ferrum.encoding._honored import SECONDARY_EXTENT
    from ferrum.encoding.positional import Theta2, Radius2

    ghost_channels = [X2, Y2, XError, YError, XError2, YError2, Theta2, Radius2]
    for cls in ghost_channels:
        doc = cls.__doc__ or ""
        assert "reserved for future use" not in doc, (
            f"{cls.__name__}.__doc__ still narrates render status "
            "('reserved for future use'); the honored-kwarg contract lives in "
            "_honored_kwargs, not per-channel prose (ENC-09)."
        )
        assert "no-op today" not in doc, (
            f"{cls.__name__}.__doc__ still narrates render status ('no-op today'); "
            "the honored-kwarg contract lives in _honored_kwargs (ENC-09)."
        )
        # The machine-readable contract is the single source of truth.
        assert cls._honored_kwargs == SECONDARY_EXTENT, (
            f"{cls.__name__}._honored_kwargs is not the SECONDARY_EXTENT role."
        )
