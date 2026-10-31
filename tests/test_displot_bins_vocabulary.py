"""The ``displot(bins=)`` vocabulary (#193, wave 4 task 9).

``bins="fd"`` produced Sturges binning with no signal -- and so did every
other rule string, valid (``"freedman_diaconis"``) or nonsense (``"auto"``,
``"stone"``). So did a float, and so did a ``numpy`` integer. ``bins=True``
produced a one-bin histogram (``bool`` subclasses ``int``), ``bins=-3``
leaked a raw ``OverflowError``, and ``bins=0`` raised a message naming the
``Bin`` transform rather than the ``displot`` call the user wrote.

The whole cluster came from one line -- a ``bins if isinstance(bins, int)
else None`` pre-filter in ``src/ferrum/plots/distribution.py`` -- in front of
an engine that already implemented Freedman-Diaconis and already refused
unknown spellings by name: ``crates/ferrum-core/src/transform/bin_mode.rs``'s
``parse_bin_axis``, reachable today through ``Bin2D(bins_x=)`` and
``Kde(count_scale=)``. The fix routes the 1-D door through that same parser
(``Bin(bins=)``), so the vocabulary has exactly one owner.

This module pins that vocabulary row by row, including the four rows that
used to be silent and the two that used to raise untyped, plus the two
hazards the routing had to clear first: every mode survives the
``spec -> wire -> spec`` round trip, and a rule-derived mode nices against
the same target whether the extent is pinned facet-wide or derived per panel.

Scope note (`tests/marks/test_boxen_palette.py`'s convention): this is the
vocabulary's own feature module, not a findings-ID file. The separate
channel-object-``hue=`` regression that shipped in the same commit is pinned
in `TestChannelObjectHue` below, deliberately apart from the binning classes.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._core import ChartSpec
from tests._svg_extents import assert_bars_start_on_the_zero_tick
from tests._svg_rects import data_bar_rects, polylines

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_df():
    """200 standard-normal draws -- enough spread for a non-degenerate IQR.

    Freedman-Diaconis falls back to Sturges whenever the IQR is zero, so a
    fixture without real spread would make every FD assertion below pass for
    the wrong reason. With this seed the two rules disagree (12 bins vs 9),
    which is what makes "FD fired" a checkable fact.
    """
    rng = np.random.default_rng(0)
    return pl.DataFrame({"v": rng.normal(size=200), "g": ["a", "b"] * 100})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bin_wire(chart) -> dict:
    """Return the single ``bin`` transform's wire dict from *chart*'s spec."""
    spec = json.loads(chart.to_spec().to_json())
    bins = [t for t in spec.get("transforms", []) if t.get("type") == "bin"]
    assert len(bins) == 1, f"expected exactly one bin transform, got {bins}"
    return bins[0]


def _kde_wire(chart) -> dict:
    spec = json.loads(chart.to_spec().to_json())
    kdes = [t for t in spec.get("transforms", []) if t.get("type") == "kde"]
    assert len(kdes) == 1, f"expected exactly one kde transform, got {kdes}"
    return kdes[0]


def _n_bars(chart) -> int:
    return len(data_bar_rects(chart.to_svg()))


def _bar_widths(chart) -> set[float]:
    return {round(float(r["width"]), 3) for r in data_bar_rects(chart.to_svg())}


def _x_tick_labels(svg: str) -> list[str]:
    """Return the sorted distinct x-axis tick labels.

    The x-axis labels are the numeric ``<text>`` elements sharing the largest
    ``y`` (the bottom of the panel); a faceted chart repeats them per panel,
    so the set -- not the list -- is what two charts can be compared on.
    """
    texts = re.findall(r'<text[^>]*x="([\d.]+)"[^>]*y="([\d.]+)"[^>]*>(-?[\d.]+)</text>', svg)
    assert texts, "chart rendered no numeric axis labels"
    bottom = max(float(y) for _, y, _ in texts)
    return sorted({label for _, y, label in texts if float(y) == bottom})


def _value_axis_map(svg: str):
    """Return a `screen_y -> axis value` function fitted from the y-axis labels.

    The y-axis tick labels all share one ``x``; they are the largest such
    group of numeric ``<text>`` elements. Two of them fix the linear map,
    which is what lets an assertion talk about *counts* instead of pixels --
    necessary here because two charts with different binning have different
    y-domains, so raw screen positions are not comparable between them.
    """
    texts = re.findall(r'<text[^>]*x="([\d.]+)"[^>]*y="([\d.]+)"[^>]*>(-?[\d.]+)</text>', svg)
    by_x: dict[str, list[tuple[float, float]]] = {}
    for x, y, label in texts:
        by_x.setdefault(x, []).append((float(y), float(label)))
    column = max(by_x.values(), key=len)
    column.sort()
    (y0, v0), (y1, v1) = column[0], column[-1]
    assert y1 != y0, "y-axis tick labels must span more than one position"
    slope = (v1 - v0) / (y1 - y0)
    return lambda y: v0 + (y - y0) * slope


def _overlay_peak_count(svg: str) -> float:
    """Return the kde overlay curve's peak, in the value axis's own units.

    Vertices come from ``tests/_svg_rects``' shared ``<polyline>`` parser,
    which this module's other probe (``data_bar_rects``) already binds to --
    rolling a second regex here was how the duplication that module exists to
    stop got one more copy in the same change that created it.
    """
    curves = polylines(svg)
    assert len(curves) == 1, f"expected one overlay curve, got {len(curves)}"
    _stroke, points = curves[0]
    return _value_axis_map(svg)(min(y for _x, y in points))


# ---------------------------------------------------------------------------
# Spec 4.5, row by row.
# ---------------------------------------------------------------------------


class TestBinsVocabularyAccepts:
    def test_sturges_and_none_are_the_default_rule(self, normal_df):
        """`"sturges"`, `None` and the default all bin identically.

        Pins the two §4.5 rows declared unchanged. Sturges stays the wire's
        *absence* of a rule key, so a pre-#193 payload keeps its exact bytes.
        """
        default = fm.displot(normal_df, x="v")
        for bins in ("sturges", None):
            chart = fm.displot(normal_df, x="v", bins=bins)
            assert chart.to_svg() == default.to_svg(), f"bins={bins!r} moved the default render"
            wire = _bin_wire(chart)
            assert "bin_rule" not in wire, f"bins={bins!r} must not spell a rule: {wire}"
            assert "bin_count" not in wire and "bin_width" not in wire, wire

    @pytest.mark.parametrize("spelling", ["fd", "freedman_diaconis"])
    def test_freedman_diaconis_actually_bins_by_freedman_diaconis(self, normal_df, spelling):
        """Both FD spellings reach the engine's FD arm.

        The discriminating fact is the bin count: FD gives 12 bins on this
        fixture where Sturges gives 9. Before the fix both spellings rendered
        the 9-bin Sturges histogram and said nothing.
        """
        chart = fm.displot(normal_df, x="v", bins=spelling)
        assert _bin_wire(chart)["bin_rule"] == "freedman_diaconis"
        sturges_bars = _n_bars(fm.displot(normal_df, x="v", bins="sturges"))
        assert sturges_bars == 9, (
            f"fixture drifted: Sturges no longer gives 9 bars ({sturges_bars})"
        )
        assert _n_bars(chart) == 12, (
            f"bins={spelling!r} did not bin by Freedman-Diaconis: got {_n_bars(chart)} bars"
        )

    def test_int_fixes_the_bin_count(self, normal_df):
        chart = fm.displot(normal_df, x="v", bins=7)
        wire = _bin_wire(chart)
        assert wire["bin_count"] == 7
        assert "bin_rule" not in wire, f"an int must stay a flat count on the wire: {wire}"
        assert _n_bars(chart) == 7

    def test_numpy_integer_is_a_bin_count_not_a_silent_degrade(self, normal_df):
        """`np.int64(9)` must bin like `int(9)`.

        It is not a Python `int`, so the old `isinstance(bins, int)` filter
        dropped it to `None` (Sturges). The parser now asks `__index__`, the
        protocol every numpy integer implements.
        """
        chart = fm.displot(normal_df, x="v", bins=np.int64(11))
        assert _bin_wire(chart)["bin_count"] == 11
        assert chart.to_svg() == fm.displot(normal_df, x="v", bins=11).to_svg()

    def test_float_is_a_bin_width_not_a_degrade(self, normal_df):
        """A float is a bin WIDTH (numpy / `Bin2D` / `Kde` parity).

        Halving the width doubles the bin count and halves each bar's
        rendered width -- the property a bin *count* reading could not
        produce, and which the old silent degrade to Sturges did not produce
        either (it ignored the number entirely).
        """
        half = fm.displot(normal_df, x="v", bins=0.5)
        quarter = fm.displot(normal_df, x="v", bins=0.25)
        assert _bin_wire(half)["bin_width"] == 0.5
        assert _n_bars(half) == 10
        assert _n_bars(quarter) == 2 * _n_bars(half)
        (half_w,) = _bar_widths(half)
        (quarter_w,) = _bar_widths(quarter)
        assert abs(quarter_w * 2 - half_w) < 0.05, (
            f"a halved bin width must halve the rendered bar width: {quarter_w} vs {half_w}"
        )


class TestBinsVocabularyRefuses:
    """Every malformed `bins=` refuses as a `ValueError` naming `displot`.

    `pytest.raises(ValueError)` is itself discriminating for two of these
    rows: `bins=-3` used to raise `OverflowError` and a channel-object-style
    wrong type used to be swallowed, and neither is a `ValueError`.
    """

    @pytest.mark.parametrize("bad", ["auto", "scott", "sqrt", "rice", "doane", "stone"])
    def test_unknown_rule_string_refuses_naming_the_accepted_set(self, normal_df, bad):
        with pytest.raises(ValueError) as exc:
            fm.displot(normal_df, x="v", bins=bad).to_svg()
        message = str(exc.value)
        assert message.startswith("displot:"), f"refusal must name the caller's entry: {message}"
        assert f"unknown bins value '{bad}'" in message
        assert "'sturges'|'fd'|int|float" in message

    @pytest.mark.parametrize("bad", [True, False])
    def test_bool_refuses_rather_than_requesting_one_bin(self, normal_df, bad):
        """`bins=True` used to render a 1-bin histogram: `bool` subclasses `int`."""
        with pytest.raises(ValueError) as exc:
            fm.displot(normal_df, x="v", bins=bad).to_svg()
        message = str(exc.value)
        assert message.startswith("displot: bins must not be a bool"), message
        assert str(bad) in message, f"refusal must name the observed value: {message}"

    @pytest.mark.parametrize("bad", [0, -3, -1])
    def test_zero_and_negative_int_refuse_typed_naming_the_bound(self, normal_df, bad):
        """No raw `OverflowError`, and no reinterpretation as a bin width.

        `bins=-3` used to fail the unsigned extraction and land in the float
        arm as a bin width of -3; `bins=0` used to name the private `Bin`
        transform.
        """
        with pytest.raises(ValueError) as exc:
            fm.displot(normal_df, x="v", bins=bad).to_svg()
        message = str(exc.value)
        assert message.startswith("displot: bins as an int must be a bin count >= 1"), message
        assert f"got {bad}" in message

    @pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
    def test_nonpositive_or_nonfinite_float_refuses(self, normal_df, bad):
        with pytest.raises(ValueError) as exc:
            fm.displot(normal_df, x="v", bins=bad).to_svg()
        assert str(exc.value).startswith(
            "displot: bins as a float must be a positive finite bin width"
        ), str(exc.value)

    @pytest.mark.parametrize("bad", [[0, 1, 2], (1, 2), {"n": 3}, object()])
    def test_wrong_type_refuses_naming_the_accepted_set(self, normal_df, bad):
        with pytest.raises(ValueError) as exc:
            fm.displot(normal_df, x="v", bins=bad).to_svg()
        assert str(exc.value) == "displot: bins must be 'sturges'|'fd'|int|float"


# ---------------------------------------------------------------------------
# Clean inputs are byte-identical (spec §7), measured rather than assumed.
# ---------------------------------------------------------------------------


#: SHA-256 of the rendered SVG for each path §7 declares unchanged, MEASURED ON
#: THE PRE-FIX BUILD -- the sources reverted by file copy and the extension
#: rebuilt -- so these are a record of the old behavior, not a re-baseline of
#: the new one.
#:
#: `hue_str_stack` (`x="v", hue="g", multiple="stack"`) was in this table and
#: is not anymore. It is not re-baselined: a SIBLING task in the same wave
#: (#194) deliberately changed that render -- `multiple="stack"` now bins every
#: group against one global extent, so the bars finally share edges and stack.
#: This table's claim is "the bins-vocabulary change moved nothing", and that
#: claim is no longer observable on a path another change moves on purpose.
#: The stacked path's new geometry is pinned by
#: `tests/test_displot_multiple_shared_edges.py`, which asserts what the bars
#: must do rather than a digest of what they happened to do.
#:
#: The histogram rows (`default`, `sturges`, `none`, `int7`, `kde_overlay_int7`,
#: `hue_str_hist`) are re-measured with the bars grown from the 0 tick, which
#: moves only their `<rect>` elements; the `hue_str_kde` row draws no bars and
#: is never re-measured. Each re-measured digest encodes two changes:
#: the bins vocabulary leaving the path untouched, and the bars grown from the
#: 0 tick, which `test_clean_histogram_paths_start_on_the_zero_tick` asserts
#: directly.
#:
#: This table pins the binning kinds only: `kind="ecdf"` counts each
#: observation itself rather than binning, so its render is not a claim
#: this table's "the bins vocabulary leaves the path untouched" makes. The
#: ecdf render itself is pinned by
#: `tests/test_log_axis_statistics.py::test_the_ecdf_steps_at_every_observation`,
#: and its inert `bins=` warning is pinned in
#: `tests/test_displot_requires_x.py::TestInertBinsWarns`.
_PRE_FIX_SVG_DIGESTS = {
    "default": "75f92caa6814f72f5890ef1dd938a0d2e63ca5ceb9bb58f2e2e93afc4c75f926",
    "sturges": "75f92caa6814f72f5890ef1dd938a0d2e63ca5ceb9bb58f2e2e93afc4c75f926",
    "none": "75f92caa6814f72f5890ef1dd938a0d2e63ca5ceb9bb58f2e2e93afc4c75f926",
    "int7": "73721a813b2c3dfc4c3da5b9f99961fc4962dfd10e4a9b87f94cd2caf122b675",
    "kde_overlay_int7": "b17a16a139e5085bd66d58e4725480dec832eb2dd78db55cc606b377a397747b",
    "hue_str_hist": "3ac4841424604764fc17557a1bf1fbb94977a517fd77fa6b24a5e589dfd6c887",
    "hue_str_kde": "4b5bf25e46b4335ad48c2e3f5668f59cbd0a039835e730217c02acfe93461c63",
}

_CLEAN_PATHS = {
    "default": dict(x="v"),
    "sturges": dict(x="v", bins="sturges"),
    "none": dict(x="v", bins=None),
    "int7": dict(x="v", bins=7),
    "kde_overlay_int7": dict(x="v", kind="hist", kde=True, stat="count", bins=7),
    "hue_str_hist": dict(x="v", hue="g"),
    "hue_str_kde": dict(x="v", hue="g", kind="kde"),
}


_CLEAN_HISTOGRAM_PATHS = ("default", "sturges", "none", "int7", "kde_overlay_int7", "hue_str_hist")


@pytest.mark.parametrize("name", _CLEAN_HISTOGRAM_PATHS)
def test_clean_histogram_paths_start_on_the_zero_tick(normal_df, name):
    """The geometry the re-measured digests pin: every bar grows from 0."""
    assert_bars_start_on_the_zero_tick(fm.displot(normal_df, **_CLEAN_PATHS[name]).to_svg())


@pytest.mark.parametrize("name", sorted(_CLEAN_PATHS))
def test_clean_paths_render_byte_identically_to_the_pre_fix_build(normal_df, name):
    """Nothing §7 declares unchanged moved.

    An `int`, `None`, the default rule, the bare-string `hue=` paths and the
    ecdf/overlay paths all reach the same `BinMode` through a different
    parameter than before, so "unchanged" is a claim about the whole
    Python-to-pixels route, not just about the wire.
    """
    svg = fm.displot(normal_df, **_CLEAN_PATHS[name]).to_svg()
    digest = hashlib.sha256(svg.encode()).hexdigest()
    assert digest == _PRE_FIX_SVG_DIGESTS[name], f"{name} render moved: {digest}"


# ---------------------------------------------------------------------------
# Hazard 1: every mode survives spec -> wire -> spec.
# ---------------------------------------------------------------------------


class TestBinWireRoundTrip:
    @pytest.mark.parametrize(
        "bins,expected",
        [
            ("sturges", {}),
            ("fd", {"bin_rule": "freedman_diaconis"}),
            ("freedman_diaconis", {"bin_rule": "freedman_diaconis"}),
            (7, {"bin_count": 7}),
            (0.5, {"bin_width": 0.5}),
        ],
    )
    def test_every_mode_states_itself_on_the_wire_and_reads_back(self, normal_df, bins, expected):
        """`to_dict()`/`to_json()` must state the binning the chart renders.

        Before the fix, `BinMode::FreedmanDiaconis` mapped to *neither* flat
        numeric key -- the same shape Sturges serializes as -- so an FD chart's
        JSON described a Sturges chart, and a round trip through it silently
        became one.
        """
        chart = fm.displot(normal_df, x="v", bins=bins)
        wire = _bin_wire(chart)
        for key in ("bin_rule", "bin_count", "bin_width"):
            assert wire.get(key) == expected.get(key), f"{key} on the wire: {wire}"

        spec = chart.to_spec()
        restored = ChartSpec.from_json(spec.to_json())
        assert restored.to_json() == spec.to_json(), "spec -> wire -> spec is not stable"
        assert restored == spec


# ---------------------------------------------------------------------------
# Hazard 2: the nicing target agrees between the facet pin and each panel.
# ---------------------------------------------------------------------------


class TestFacetedRuleBinningAgrees:
    def test_faceted_fd_shares_the_unfaceted_binning_range(self, normal_df):
        """Rule-derived modes nice against the Sturges count on both paths.

        The facet-wide extent pin and each panel's own binning ask the same
        question (`BinMode::nice_target_count`). Were the pin to nice against
        FD's own count instead, the pinned range would stop matching the range
        the unfaceted path derives, and the value axis would move. Compared via
        the rendered x tick labels because screen geometry is not comparable
        across panels of different widths.

        The byte-level companion -- the pinned and auto paths producing the
        identical `bin_start` values under FD -- is pinned in Rust by
        `fd_niced_global_extent_matches_the_apply_path_edges` (`bin.rs`), which
        can read the transform output directly.
        """
        faceted = fm.displot(normal_df, x="v", col="g", bins="fd")
        assert _bin_wire(faceted)["bin_rule"] == "freedman_diaconis"
        plain = fm.displot(normal_df, x="v", bins="fd")
        assert _x_tick_labels(faceted.to_svg()) == _x_tick_labels(plain.to_svg()), (
            "faceted FD binning covers a different range than the unfaceted binning"
        )


# ---------------------------------------------------------------------------
# The kde overlay tracks the bars' ACTUAL binning.
# ---------------------------------------------------------------------------


class TestOverlayTracksTheRealBinning:
    def test_overlay_count_scale_carries_the_rule_not_a_downgrade(self, normal_df):
        """`count_scale` carries `bins` through, rule and all.

        A helper used to exist purely to downgrade `"fd"` to `"sturges"` here,
        because the bars could not honor FD. They can now, so the overlay
        states the same rule the bars were binned by.
        """
        chart = fm.displot(normal_df, x="v", kind="hist", kde=True, stat="count", bins="fd")
        assert _kde_wire(chart)["count_scale"] == {"kind": "freedman_diaconis"}
        assert _bin_wire(chart)["bin_rule"] == "freedman_diaconis"

    def test_overlay_curve_is_scaled_by_the_rules_own_bin_width(self, normal_df):
        """The rendered curve peak scales with the bin width the bars used.

        Count-scaling multiplies the density by `n * bin_width`, so FD's 12
        bins over the same niced range must put the curve at 9/12 of the peak
        Sturges' 9 bins put it at. Under the old downgrade the two curves were
        *identical* (both scaled by the Sturges width), so a ratio of 1.0 is
        exactly the failure this discriminates.
        """
        fd_svg = fm.displot(
            normal_df, x="v", kind="hist", kde=True, stat="count", bins="fd"
        ).to_svg()
        sturges_svg = fm.displot(
            normal_df, x="v", kind="hist", kde=True, stat="count", bins="sturges"
        ).to_svg()
        ratio = _overlay_peak_count(fd_svg) / _overlay_peak_count(sturges_svg)
        assert abs(ratio - 9 / 12) < 0.02, (
            f"overlay curve did not follow the bars' FD bin width: peak ratio {ratio:.3f}, "
            "expected 0.75 (1.0 means the overlay was scaled by the Sturges width)"
        )


# ---------------------------------------------------------------------------
# The same vocabulary governs the ecdf path and the mark-level entry.
# ---------------------------------------------------------------------------


class TestVocabularyReachesEveryBinningEntry:
    def test_ecdf_still_validates_the_rule_but_it_is_now_inert(self, normal_df):
        """`kind="ecdf"` counts each observation itself through
        `transform_window`, never building a `Bin`, so a stated rule is
        validated by the same vocabulary -- a malformed one still refuses --
        but warns inert rather than feeding a transform."""
        with pytest.warns(UserWarning) as record:
            chart = fm.displot(normal_df, x="v", kind="ecdf", bins="fd")
        messages = [str(w.message) for w in record]
        assert any(
            m.startswith("displot(kind='ecdf', bins='fd')") and "inert" in m for m in messages
        ), messages
        spec = json.loads(chart.to_spec().to_json())
        assert not any(t["type"] == "bin" for t in spec.get("transforms", []))
        with pytest.raises(ValueError) as exc:
            fm.displot(normal_df, x="v", kind="ecdf", bins="auto")
        assert str(exc.value).startswith("displot: unknown bins value 'auto'")

    def test_mark_histogram_accepts_the_rule_and_bin_refuses_a_double_answer(self, normal_df):
        """`mark_histogram(bins=)` is the route `displot` takes; it works directly too.

        `Bin` owns the mutual exclusion, so a caller who answers the binning
        question twice is refused rather than having one answer dropped.
        """
        chart = fm.Chart(normal_df).mark_histogram(bins="fd").encode(x="v")
        assert _bin_wire(chart)["bin_rule"] == "freedman_diaconis"
        with pytest.raises(ValueError) as exc:
            fm.Bin("v", bins="fd", bin_count=10)
        assert "mutually exclusive" in str(exc.value)


# ---------------------------------------------------------------------------
# Rider (#193's adjudicated second defect): a channel-object `hue=`.
# ---------------------------------------------------------------------------


class TestChannelObjectHue:
    """`hue=fm.Color("g")` is a documented spelling; every branch must unwrap it.

    `kind="kde"` raised `ValueError: groupby must be None, a str, or a
    list[str]`, because the branch passed the channel object straight into the
    transform's `groupby=`. The same object also reached `Stack(by=)`/
    `Dodge(by=)` under `multiple=`, where it raised `TypeError: Object of type
    Color is not JSON serializable`. `displot` now resolves the bare column
    name once, up front, and the color *encoding* keeps the channel.

    This matters more since #130 landed in this wave: an untyped `fm.Color("g")`
    now reads as nominal, so the channel spelling is the one the guidance
    points at.
    """

    @pytest.mark.parametrize("kind", ["hist", "kde", "ecdf", "rug"])
    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill", "dodge"])
    def test_channel_hue_renders_on_every_kind_and_mode(self, normal_df, kind, multiple):
        svg = fm.displot(normal_df, x="v", hue=fm.Color("g"), kind=kind, multiple=multiple).to_svg()
        assert "NaN" not in svg and "Infinity" not in svg

    @pytest.mark.parametrize("kind", ["hist", "kde", "ecdf", "rug"])
    def test_channel_hue_matches_the_bare_string_spelling(self, normal_df, kind):
        """Both spellings of the same hue must produce the same chart.

        Discriminating against a fix that unwraps the name but loses the
        grouping: a dropped `groupby=` would still render, just with one
        pooled mark instead of one per level.
        """
        typed = fm.displot(normal_df, x="v", hue=fm.Color("g"), kind=kind).to_svg()
        bare = fm.displot(normal_df, x="v", hue="g", kind=kind).to_svg()
        assert typed == bare

    def test_channel_hue_survives_the_kde_overlay(self, normal_df):
        chart = fm.displot(normal_df, x="v", hue=fm.Color("g"), kind="hist", kde=True)
        assert _kde_wire(chart)["groupby"] == ["g"]
