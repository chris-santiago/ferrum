import re

import polars as pl
import pytest

from ferrum.marks.base import MarkBase


def test_markbase_accepts_valid_kwargs():
    m = MarkBase("point", size=100, stroke="#ff0000", opacity=0.5)
    assert m._kwargs == {"size": 100, "stroke": "#ff0000", "opacity": 0.5}


def test_markbase_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unknown keyword"):
        MarkBase("point", squiggly=True)


def test_to_mark_kwargs_dict_filters_to_style_only():
    m = MarkBase("smooth", size=100, method="loess", bandwidth=0.5)
    d = m.to_mark_kwargs_dict()
    assert d == {"size": 100}   # method and bandwidth go to transforms, not style


def test_deferred_mark_error_for_9_plus_mark():
    from ferrum.marks import deferred_mark_error
    e = deferred_mark_error("arc")
    # Phase 9 is closed; these marks are now planned for Phase 11+.
    assert "Phase 11+" in str(e)


def test_phase_8b_marks_set_is_empty_after_subbatch_f():
    from ferrum.marks import PHASE_8B_MARKS
    # Sub-batch E shipped composites (boxplot/errorbar/errorband/ribbon) and
    # Sub-batch F shipped heavy-stat marks (contour/violin/qq/raster/hex/swarm/function).
    # All Phase 8b marks are now implemented; the deferred set is empty.
    assert PHASE_8B_MARKS == frozenset()


def test_desugar_density_returns_area_with_kde_transform():
    from ferrum.marks.statistical import desugar_density
    from ferrum import Kde
    mark, transforms, remap = desugar_density("price")
    assert mark == "area"
    assert len(transforms) == 1 and isinstance(transforms[0], Kde)
    # Kde produces ("value", "density") columns; both x and y are remapped.
    assert remap == {"x": "value", "y": "density"}


def test_desugar_histogram_returns_bar_with_bin_transform():
    from ferrum.marks.statistical import desugar_histogram
    from ferrum import Bin
    mark, transforms, remap = desugar_histogram("price", bin_count=20)
    assert mark == "bar"
    assert isinstance(transforms[0], Bin)
    assert remap == {"x": "bin_start", "x2": "bin_end", "y": "count"}


def test_desugar_smooth_with_ci_returns_layered_tuple():
    """Phase 8b: ci= no longer warns; it returns the layered (__layered__,
    transforms, _, _, layers) 5-tuple emitting a ribbon CI band + line."""
    import warnings
    from ferrum._warn import reset_warnings
    from ferrum.marks.statistical import desugar_smooth

    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = desugar_smooth("x_col", "y_col", ci=0.95)
    # No ci-deferral warning anymore.
    assert not any("Phase 8b" in str(wi.message) and "ci" in str(wi.message).lower() for wi in w)
    # 5-tuple layered output, ribbon + line layers.
    assert isinstance(result, tuple) and len(result) == 5
    assert result[0] == "__layered__"
    layers = result[4]
    assert [layer["mark"] for layer in layers] == ["ribbon", "line"]


# ---------------------------------------------------------------------------
# Task 36: deferred-mark NotImplementedError parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,phase,extra_args", [
    # Phase 9 closed. arc/geoshape/label remain deferred to Phase 11+.
    ("mark_arc", "11", ()),
    ("mark_geoshape", "11", ()),
    ("mark_label", "11", ()),
])
def test_deferred_mark_methods_raise_with_phase_pointer(method, phase, extra_args):
    df = pl.DataFrame({"a": [1]})
    from ferrum import Chart
    c = Chart(df).encode(x="a", y="a")
    with pytest.raises(NotImplementedError, match=f"Phase {phase}"):
        getattr(c, method)(*extra_args)


# ---------------------------------------------------------------------------
# Task 36: e2e log-scale roundtrip — load-bearing test
# ---------------------------------------------------------------------------

def test_encode_with_explicit_log_scale_renders_log_axis():
    """Proves EncodingSpec.scale roundtrip (Task 3): LogScale flows to renderer.

    With an explicit LogScale(domain=[1, 1000]), the x-axis ticks are
    logarithmically spaced. Without it, the renderer defaults to linear,
    producing evenly-spaced ticks across the data range (400, 500, 600, ...).
    We verify the log path by checking that a logarithmically-intermediate
    value like "10" appears as a tick label (it would never appear on a
    linear scale over this range), and that purely linear-spaced ticks
    (400, 500, 600 ...) are absent from the axis text labels.
    """
    from ferrum import Chart, X, LogScale
    from ferrum._warn import reset_warnings

    reset_warnings()
    df = pl.DataFrame({
        "x": [1.0, 10.0, 100.0, 1000.0],
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    svg = Chart(df).mark_point().encode(
        x=X("x", scale=LogScale(domain=[1.0, 1000.0], range=[0.0, 600.0], base=10.0)),
        y="y",
    ).show_svg()

    # Extract text-element labels from the SVG (strip font/base64 blobs).
    text_labels = re.findall(r"<text[^>]*>([^<]+)</text>", svg)

    # The x-axis should contain tick labels at log-spaced positions.
    # "10" appears only under a log scale over this domain (1–1000).
    assert "10" in text_labels, (
        f"Log-scale x axis missing '10' tick label — "
        f"explicit ScaleLog was likely dropped. x labels: {text_labels}"
    )

    # Linear-scale ticks over [1, 1000] would produce labels like 400, 500, 600.
    # Under a log scale none of these are tick positions.
    linear_spill = {"400", "500", "600"} & set(text_labels)
    assert not linear_spill, (
        f"Found linear-scale tick labels {linear_spill} in log-scale render — "
        f"explicit ScaleLog was silently dropped. x labels: {text_labels}"
    )


# ---------------------------------------------------------------------------
# Test 1: stat-mark order independence (Phase 8a final review)
# ---------------------------------------------------------------------------

def test_mark_smooth_then_encode_resolves_correctly():
    """Confirms _pending_stat_mark machinery in Chart works regardless of method order."""
    from ferrum import Chart
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 4.0, 9.0, 16.0]})
    # Order: mark BEFORE encode (the e2e example pattern)
    c = Chart(df).mark_smooth().encode(x="a", y="b")
    spec = c.to_spec()
    assert spec.mark == "line"  # smooth desugared to line + Smooth transform
    assert any(t.__class__.__name__ == "Smooth" for t in spec.transforms)


def test_mark_density_then_encode_resolves_correctly():
    from ferrum import Chart
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    c = Chart(df).mark_density().encode(x="a")
    spec = c.to_spec()
    assert spec.mark == "area"
    assert any(t.__class__.__name__ == "Kde" for t in spec.transforms)


def test_mark_histogram_then_encode_resolves_correctly():
    from ferrum import Chart
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    c = Chart(df).mark_histogram().encode(x="a")
    spec = c.to_spec()
    assert spec.mark == "bar"
    assert any(t.__class__.__name__ == "Bin" for t in spec.transforms)
