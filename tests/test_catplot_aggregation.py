"""``catplot(kind="point"/"bar")`` aggregation coverage (batch F, F-L09-02/03).

Before this feature landed, both estimating kinds drew ONE MARK PER ROW -- a
200-row frame over 4 categories rendered 200 overplotted circles or bars -- and
``ci=``/``n_boot=`` raised "not yet supported; the Summary transform is not
wired into catplot" for any non-default value. This module pins the shipped
contract: one mean estimate per (category x hue) group, drawn against a seeded
bootstrap CI whisker, with the whisker UNDER the marker for ``kind="point"``
and OVER the bar for ``kind="bar"``.

Assertions read the RENDERED SVG, not the spec dict: the defect this closes was
visible geometry (200 marks where 4 belong), and a spec-shaped assertion would
have passed against the broken output. Net-new feature coverage lives in its
own module per the test-homing convention in CLAUDE.md.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm

from _svg_rects import data_bar_rects, svg_elements

# The whisker is a `rule` mark stroked with the theme's `label` token. Axis
# domain/tick lines share that color but render at stroke-width 1, and
# gridlines at 0.5 -- 1.5 is the default theme's rule width, so the pair
# (color, width) isolates the CI whiskers from every other <line> in the SVG.
_LABEL_STROKE = "#6b7280"
_WHISKER_STROKE_WIDTH = "1.5"

# Facet header bands are filled rects too, and `data_bar_rects` only excludes
# the plot background -- so the bar counts filter this chrome fill out as well.
_FACET_BAND_FILL = "#ede9e3"


def whisker_lines(svg: str) -> list[dict[str, str]]:
    """Return the attribute dict of every rendered CI-whisker ``<line>``."""
    return svg_elements(
        svg,
        "line",
        lambda a: (
            a.get("stroke") == _LABEL_STROKE and a.get("stroke-width") == _WHISKER_STROKE_WIDTH
        ),
    )


def whisker_spans(svg: str) -> list[tuple[float, float, float]]:
    """Return ``(x, y_start, y_end)`` for every whisker, rounded for comparison."""
    return [
        (round(float(a["x1"]), 3), round(float(a["y1"]), 3), round(float(a["y2"]), 3))
        for a in whisker_lines(svg)
    ]


def first_index(svg: str, pattern: str) -> int:
    """Document position of the first element matching *pattern*.

    Paint order is document order, so comparing these positions is how the
    whisker-under-marker / whisker-over-bar contract is read off the render.
    """
    match = re.search(pattern, svg)
    assert match is not None, f"no element matching {pattern!r} in the rendered SVG"
    return match.start()


_WHISKER_ELEMENT = rf'<line[^/]*stroke="{_LABEL_STROKE}"[^/]*stroke-width="{_WHISKER_STROKE_WIDTH}"'
_DATA_MARK_FILL = r'fill="#2563eb"'


def estimator_bars(svg: str) -> list[dict[str, str]]:
    """Return the mean-bar ``<rect>`` elements, excluding panel/facet chrome."""
    return [r for r in data_bar_rects(svg) if r["fill"] != _FACET_BAND_FILL]


def marker_circles(svg: str) -> list[dict[str, str]]:
    """Return the mean-marker ``<circle>`` elements, excluding legend swatches.

    Legend swatches are drawn outside the plot rect; both marker and swatch are
    plain circles, so they are separated by x position against the rightmost
    whisker (the plot area always ends left of the legend band).
    """
    circles = svg_elements(svg, "circle", lambda a: "cx" in a)
    whisker_x = [float(a["x1"]) for a in whisker_lines(svg)]
    if not whisker_x:
        return circles
    plot_right = max(whisker_x)
    return [c for c in circles if float(c["cx"]) <= plot_right + 1.0]


@pytest.fixture
def cat_df() -> pl.DataFrame:
    """200 rows, 4 categories x 2 hue levels, deterministic values.

    ``cat`` and ``grp`` are deliberately NOT collinear (``grp`` switches every
    4 rows, ``cat`` every row), so every one of the 8 (category, hue) cells is
    populated -- a collinear frame would silently reduce the hue tests to 4
    groups and hide a dodge defect.
    """
    rng = np.random.default_rng(0)
    cats = ["a", "b", "c", "d"]
    rows = [
        (cats[i % 4], ["X", "Y"][(i // 4) % 2], float(rng.normal(5 + (i % 4), 1)))
        for i in range(200)
    ]
    return pl.DataFrame(rows, schema=["cat", "grp", "val"], orient="row")


# ---------------------------------------------------------------------------
# One estimate per group, not one mark per row
# ---------------------------------------------------------------------------


class TestOneMarkPerGroup:
    def test_point_draws_one_marker_per_category(self, cat_df):
        """4 categories over 200 rows -> 4 markers and 4 whiskers, not 200."""
        svg = fm.catplot(cat_df, x="cat", y="val", kind="point").to_svg()
        assert len(marker_circles(svg)) == 4
        assert len(whisker_lines(svg)) == 4

    def test_bar_draws_one_bar_per_category(self, cat_df):
        """4 categories over 200 rows -> 4 bars and 4 whiskers, not 200."""
        svg = fm.catplot(cat_df, x="cat", y="val", kind="bar").to_svg()
        assert len(estimator_bars(svg)) == 4
        assert len(whisker_lines(svg)) == 4

    def test_hue_splits_the_estimate_per_category_and_level(self, cat_df):
        """4 categories x 2 hue levels -> 8 estimates, one per populated cell."""
        svg = fm.catplot(cat_df, x="cat", y="val", hue="grp", kind="point", dodge=True).to_svg()
        assert len(marker_circles(svg)) == 8
        assert len(whisker_lines(svg)) == 8

    def test_marker_sits_between_the_whisker_endpoints(self, cat_df):
        """The mean must lie inside its own CI, per category."""
        svg = fm.catplot(cat_df, x="cat", y="val", kind="point").to_svg()
        by_x = {round(float(a["x1"]), 3): a for a in whisker_lines(svg)}
        assert len(by_x) == 4
        for circle in marker_circles(svg):
            span = by_x[round(float(circle["cx"]), 3)]
            lo, hi = sorted((float(span["y1"]), float(span["y2"])))
            assert lo <= float(circle["cy"]) <= hi


# ---------------------------------------------------------------------------
# Layer order (the whisker is under the marker, over the bar)
# ---------------------------------------------------------------------------


class TestLayerOrder:
    def test_point_layers_are_whisker_then_marker(self, cat_df):
        chart = fm.catplot(cat_df, x="cat", y="val", kind="point")
        assert chart.layer_names == ["whisker", "point"]
        svg = chart.to_svg()
        # The whisker is emitted first, so the marker covers the interval's
        # midsection rather than being hidden underneath it.
        assert first_index(svg, _WHISKER_ELEMENT) < first_index(svg, r"<circle"), (
            "the mean marker must paint over its CI whisker, not under it"
        )

    def test_bar_layers_are_bar_then_whisker(self, cat_df):
        chart = fm.catplot(cat_df, x="cat", y="val", kind="bar")
        assert chart.layer_names == ["bar", "whisker"]
        svg = chart.to_svg()
        first_bar = first_index(svg, rf"<rect[^/]*{_DATA_MARK_FILL}")
        assert first_bar < first_index(svg, _WHISKER_ELEMENT), (
            "the CI whisker must paint over the bar, not under it"
        )

    def test_whisker_spans_the_interval_not_a_zero_length_stub(self, cat_df):
        """Every whisker has real extent -- a clipped or collapsed interval
        (the shape an explicit zero-anchored scale produced during design)
        would show up as a missing or zero-length line."""
        svg = fm.catplot(cat_df, x="cat", y="val", kind="bar").to_svg()
        spans = whisker_spans(svg)
        assert len(spans) == 4
        assert all(abs(y1 - y2) > 1.0 for _, y1, y2 in spans)


# ---------------------------------------------------------------------------
# ci= / n_boot= / seed= are honored
# ---------------------------------------------------------------------------


class TestBootstrapArguments:
    def test_narrower_ci_draws_shorter_whiskers(self, cat_df):
        """ci=90 is nested inside ci=95: every whisker must be shorter."""
        default = whisker_spans(fm.catplot(cat_df, x="cat", y="val", kind="point").to_svg())
        narrow = whisker_spans(fm.catplot(cat_df, x="cat", y="val", kind="point", ci=90).to_svg())
        assert len(default) == len(narrow) == 4
        for (x_d, y1_d, y2_d), (x_n, y1_n, y2_n) in zip(default, narrow):
            assert x_d == x_n
            assert abs(y1_n - y2_n) < abs(y1_d - y2_d)

    def test_n_boot_changes_the_interval(self, cat_df):
        """A different resample count draws a different (still seeded) interval."""
        default = whisker_spans(fm.catplot(cat_df, x="cat", y="val", kind="point").to_svg())
        fewer = whisker_spans(
            fm.catplot(cat_df, x="cat", y="val", kind="point", n_boot=250).to_svg()
        )
        assert default != fewer

    def test_same_seed_is_byte_identical(self, cat_df):
        a = fm.catplot(cat_df, x="cat", y="val", kind="point", seed=7).to_svg()
        b = fm.catplot(cat_df, x="cat", y="val", kind="point", seed=7).to_svg()
        assert a == b

    def test_default_seed_is_byte_deterministic(self, cat_df):
        """seed=None maps to the fixed default seed, not to entropy."""
        a = fm.catplot(cat_df, x="cat", y="val", kind="bar").to_svg()
        b = fm.catplot(cat_df, x="cat", y="val", kind="bar").to_svg()
        assert a == b

    def test_different_seed_moves_the_whiskers(self, cat_df):
        default = whisker_spans(fm.catplot(cat_df, x="cat", y="val", kind="point").to_svg())
        seeded = whisker_spans(fm.catplot(cat_df, x="cat", y="val", kind="point", seed=7).to_svg())
        assert default != seeded

    def test_seed_moves_the_interval_without_moving_the_estimate(self, cat_df):
        """Only the interval resamples; the mean stays inside every whisker.

        The markers' PIXEL positions do shift between seeds -- a different
        interval is a different value-axis domain -- so the invariant is the
        containment relation, not raw coordinates.
        """
        for seed in (None, 7):
            svg = fm.catplot(cat_df, x="cat", y="val", kind="point", seed=seed).to_svg()
            by_x = {round(float(a["x1"]), 3): a for a in whisker_lines(svg)}
            for circle in marker_circles(svg):
                span = by_x[round(float(circle["cx"]), 3)]
                lo, hi = sorted((float(span["y1"]), float(span["y2"])))
                assert lo <= float(circle["cy"]) <= hi


# ---------------------------------------------------------------------------
# hue composition: groupby + dodge move whisker and marker together
# ---------------------------------------------------------------------------


class TestHueComposition:
    def test_dodge_offsets_whisker_and_marker_identically(self, cat_df):
        """Whisker and marker carry the SAME position adjustment.

        A whisker that inherited the chart-level adjustment (or none at all)
        while the marker dodged would separate the interval from its estimate;
        parsed x positions are what discriminate that.
        """
        svg = fm.catplot(cat_df, x="cat", y="val", hue="grp", kind="point", dodge=True).to_svg()
        whisker_x = sorted(round(float(a["x1"]), 3) for a in whisker_lines(svg))
        marker_x = sorted(round(float(c["cx"]), 3) for c in marker_circles(svg))
        assert len(whisker_x) == 8
        assert whisker_x == marker_x
        assert len(set(whisker_x)) == 8, "dodge must give each hue level its own offset"

    def test_dodged_bars_carry_their_whisker_to_the_same_center(self, cat_df):
        svg = fm.catplot(cat_df, x="cat", y="val", hue="grp", kind="bar", dodge=True).to_svg()
        bar_centers = sorted(float(r["x"]) + float(r["width"]) / 2 for r in estimator_bars(svg))
        whisker_x = sorted(float(a["x1"]) for a in whisker_lines(svg))
        assert len(bar_centers) == 8
        # Half-pixel tolerance: the bar's center is derived from its rounded
        # left edge and width, the whisker's from its own rounded coordinate.
        assert bar_centers == pytest.approx(whisker_x, abs=0.05)

    def test_without_dodge_the_groups_share_the_band_center(self, cat_df):
        """dodge=False overlays the hue levels (catplot's documented default)."""
        svg = fm.catplot(cat_df, x="cat", y="val", hue="grp", kind="point").to_svg()
        whisker_x = {round(float(a["x1"]), 3) for a in whisker_lines(svg)}
        assert len(whisker_lines(svg)) == 8
        assert len(whisker_x) == 4

    def test_hue_colors_the_estimator_marks(self, cat_df):
        svg = fm.catplot(cat_df, x="cat", y="val", hue="grp", kind="bar", dodge=True).to_svg()
        fills = {r["fill"] for r in estimator_bars(svg)}
        assert len(fills) == 2, f"expected one palette color per hue level, got {sorted(fills)}"


# ---------------------------------------------------------------------------
# order= / hue_order= still act through the layered encodings
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_order_reorders_the_category_axis(self, cat_df):
        svg = fm.catplot(
            cat_df, x="cat", y="val", kind="point", order=["d", "c", "b", "a"]
        ).to_svg()
        labels = [t for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg) if t in set("abcd")]
        assert labels == ["d", "c", "b", "a"]

    def test_hue_order_reorders_the_legend(self, cat_df):
        svg = fm.catplot(
            cat_df, x="cat", y="val", hue="grp", kind="bar", hue_order=["Y", "X"]
        ).to_svg()
        labels = [t for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg) if t in {"X", "Y"}]
        assert labels == ["Y", "X"]

    def test_value_axis_keeps_the_original_column_title(self, cat_df):
        """The axis reads the user's value column, not the transform's output
        column (`mean`/`lower`), whichever layer happens to be first."""
        for kind in ("point", "bar"):
            svg = fm.catplot(cat_df, x="cat", y="val", kind=kind).to_svg()
            titles = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
            assert "val" in titles
            assert "mean" not in titles and "lower" not in titles


# ---------------------------------------------------------------------------
# The value axis: zero baseline, and a caller's explicit scale=
# ---------------------------------------------------------------------------


def text_labels(svg: str) -> list[str]:
    """Every rendered ``<text>`` string (axis ticks, titles, legend labels)."""
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


class TestValueAxis:
    def test_bar_value_axis_reaches_zero(self, cat_df):
        """The bar baseline must be ON the axis, not below the panel floor.

        `cat_df`'s means sit around 5-8, so a domain that merely spans the
        estimates would never tick 0 -- exactly what `kind="point"` shows in
        the companion assertion. The anchor comes from the renderer's
        layered-bar domain rule, which only fires while this builder leaves
        the value axis auto-inferred; fabricating an explicit scale here would
        silently destroy it, so this pins the end-to-end catplot contract, not
        just the whiskers' survival.
        """
        bar_ticks = text_labels(fm.catplot(cat_df, x="cat", y="val", kind="bar").to_svg())
        assert "0" in bar_ticks, f"bar value axis must include a zero tick; got {bar_ticks}"

        point_ticks = text_labels(fm.catplot(cat_df, x="cat", y="val", kind="point").to_svg())
        assert "0" not in point_ticks, (
            f"point draws no baseline, so its axis must stay on the estimates; got {point_ticks}"
        )

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_explicit_domain_governs_the_value_axis(self, cat_df, kind):
        """A caller's `scale=` with a domain is honored -- and honoring it must
        not cost a whisker.

        Both halves are load-bearing. Before this, the layers were rebuilt from
        bare column names and the scale was silently dropped (the axis stayed
        on the data), which was both a regression from the pre-aggregation
        single-mark path and drift from `kind="box"`, which honors it. The
        naive repair clips: an explicit scale resolves its domain from one
        column, so the count of whiskers is what proves the caller's domain
        (0-20, wider than the data) governs without truncating the estimates.
        """
        svg = fm.catplot(
            cat_df, x="cat", y=fm.Y("val", scale={"domain": [0, 20]}), kind=kind
        ).to_svg()
        labels = text_labels(svg)
        assert "20" in labels and "0" in labels, f"caller domain must govern; got {labels}"
        assert len(whisker_lines(svg)) == 4

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_explicit_domain_is_honored_under_orient_h(self, cat_df, kind):
        """The value channel is `x=` under `orient="h"`; the same scale must act."""
        svg = fm.catplot(
            cat_df, y="cat", x=fm.X("val", scale={"domain": [0, 20]}), kind=kind, orient="h"
        ).to_svg()
        assert "20" in text_labels(svg)
        assert len(whisker_lines(svg)) == 4

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_typed_scale_with_domain_is_honored(self, cat_df, kind):
        """A log scale WITH a domain is expressible: log ticks, all whiskers."""
        svg = fm.catplot(
            cat_df, x="cat", y=fm.Y("val", scale={"type": "log", "domain": [1, 20]}), kind=kind
        ).to_svg()
        assert len(whisker_lines(svg)) == 4
        # Log ticks are irregular decimals, unlike the linear axis's integers.
        assert any("." in label for label in text_labels(svg))

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_domainless_value_scale_refuses_instead_of_clipping(self, cat_df, kind):
        """A domain-less value scale is refused, not silently dropped or clipped.

        The renderer resolves an explicit domain-less scale from the single
        column the axis-defining layer encodes, which is narrower than the
        (estimate, interval) pair these kinds draw -- measured on the sibling
        `kind="box"`, that renders 1 whisker of 4. Neither silent outcome is
        acceptable, so the boundary refuses and names the route that works.
        """
        with pytest.raises(ValueError, match="without an explicit domain"):
            fm.catplot(cat_df, x="cat", y=fm.Y("val", scale={"type": "log"}), kind=kind)

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_no_scale_is_still_the_ordinary_case(self, cat_df, kind):
        """Propagation must be a no-op when no channel carries a scale."""
        spec = json.loads(fm.catplot(cat_df, x="cat", y="val", kind=kind).to_spec().to_json())
        assert all("scale" not in ly["encoding"]["y"] for ly in spec["layers"])

    # `encode={...}` is the second door onto the same channels. It is applied
    # by `_finalize_chart` AFTER the layers exist, through `Chart.encode`,
    # which writes only the chart-level encoding -- so a scale arriving that
    # way reached neither the propagation nor the refusal until the builder
    # started reading it. Both halves are pinned, because a door that skips
    # the refusal is worse than one that skips the propagation: it silently
    # drops a scale the function promises to either honor or explain.

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_encode_dict_domain_governs_the_value_axis(self, cat_df, kind):
        svg = fm.catplot(
            cat_df,
            x="cat",
            y="val",
            kind=kind,
            encode={"y": fm.Y("val", scale={"domain": [0, 20]})},
        ).to_svg()
        labels = text_labels(svg)
        assert "20" in labels and "0" in labels, f"encode= domain must govern; got {labels}"
        assert len(whisker_lines(svg)) == 4

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_encode_dict_domainless_value_scale_refuses(self, cat_df, kind):
        with pytest.raises(ValueError, match="without an explicit domain"):
            fm.catplot(
                cat_df,
                x="cat",
                y="val",
                kind=kind,
                encode={"y": fm.Y("val", scale={"type": "log"})},
            )

    def test_encode_dict_wins_over_the_y_channel(self, cat_df):
        """`encode=` is applied last by `_finalize_chart`, so it wins -- the
        propagated scale must follow that same precedence, not the reverse."""
        svg = fm.catplot(
            cat_df,
            x="cat",
            y=fm.Y("val", scale={"domain": [0, 40]}),
            kind="point",
            encode={"y": fm.Y("val", scale={"domain": [0, 20]})},
        ).to_svg()
        labels = text_labels(svg)
        assert "20" in labels and "40" not in labels, f"encode= must win over y=; got {labels}"

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_encode_dict_without_a_scale_changes_nothing(self, cat_df, kind):
        """The ordinary `encode=` call (no scale) stays a no-op on the layers."""
        spec = json.loads(
            fm.catplot(cat_df, x="cat", y="val", kind=kind, encode={"y": "val"}).to_spec().to_json()
        )
        assert all("scale" not in ly["encoding"]["y"] for ly in spec["layers"])


# ---------------------------------------------------------------------------
# Edges and refusals
# ---------------------------------------------------------------------------


class TestEdgesAndRefusals:
    def test_single_category(self, cat_df):
        one = cat_df.filter(pl.col("cat") == "a")
        svg = fm.catplot(one, x="cat", y="val", kind="point").to_svg()
        assert len(marker_circles(svg)) == 1
        assert len(whisker_lines(svg)) == 1

    def test_single_row_estimates_without_an_interval(self, cat_df):
        """One observation: the mean is that value and the bootstrap CI has
        zero width. It must render, not raise or emit NaN."""
        svg = fm.catplot(cat_df.head(1), x="cat", y="val", kind="point").to_svg()
        assert "NaN" not in svg
        assert len(marker_circles(svg)) == 1

    def test_single_row_bar(self, cat_df):
        svg = fm.catplot(cat_df.head(1), x="cat", y="val", kind="bar").to_svg()
        assert "NaN" not in svg
        assert len(estimator_bars(svg)) == 1

    def test_orient_horizontal(self, cat_df):
        """orient="h" flips both layers together (the whisker follows the bar)."""
        svg = fm.catplot(cat_df, y="cat", x="val", kind="bar", orient="h").to_svg()
        assert len(estimator_bars(svg)) == 4
        # Under CoordFlip the whiskers are horizontal: y1 == y2, x1 != x2.
        flipped = svg_elements(
            svg,
            "line",
            lambda a: (
                a.get("stroke") == _LABEL_STROKE and a.get("stroke-width") == _WHISKER_STROKE_WIDTH
            ),
        )
        assert len(flipped) == 4
        assert all(a["y1"] == a["y2"] and a["x1"] != a["x2"] for a in flipped)

    def test_faceting_estimates_per_panel(self, cat_df):
        svg = fm.catplot(cat_df, x="cat", y="val", kind="bar", col="grp").to_svg()
        assert len(estimator_bars(svg)) == 8
        assert len(whisker_lines(svg)) == 8

    @pytest.mark.parametrize("kind", ["point", "bar"])
    @pytest.mark.parametrize("ci", [0, 100, -5, 150, "wide"])
    def test_ci_outside_the_open_interval_refuses(self, cat_df, kind, ci):
        with pytest.raises(ValueError, match=r"ci=.*\(0, 100\)"):
            fm.catplot(cat_df, x="cat", y="val", kind=kind, ci=ci)

    @pytest.mark.parametrize("kind", ["point", "bar"])
    @pytest.mark.parametrize("n_boot", [0, -1, 1.5])
    def test_n_boot_below_one_refuses(self, cat_df, kind, n_boot):
        with pytest.raises(ValueError, match=r"n_boot=.*>= 1"):
            fm.catplot(cat_df, x="cat", y="val", kind=kind, n_boot=n_boot)

    def test_boolean_ci_and_n_boot_refuse(self, cat_df):
        """``bool`` subclasses ``int``, so both would otherwise be accepted --
        ``n_boot=True`` as "1 resample" and ``ci=True`` as "a 1 % interval".
        Refused by type, while the numeric 1 stays valid for both."""
        with pytest.raises(ValueError, match=r"n_boot=.*>= 1"):
            fm.catplot(cat_df, x="cat", y="val", kind="point", n_boot=True)
        with pytest.raises(ValueError, match=r"ci=.*\(0, 100\)"):
            fm.catplot(cat_df, x="cat", y="val", kind="point", ci=True)
        assert "<svg" in fm.catplot(cat_df, x="cat", y="val", kind="point", ci=1, n_boot=1).to_svg()

    @pytest.mark.parametrize("kind", ["strip", "swarm", "box", "violin", "boxen"])
    def test_ci_is_inert_on_the_non_estimating_kinds(self, cat_df, kind):
        """The six kinds that draw no interval never consumed ci=/n_boot=, so
        a value they ignore must not newly raise."""
        assert "<svg" in fm.catplot(cat_df, x="cat", y="val", kind=kind, ci=0, n_boot=0).to_svg()

    @pytest.mark.parametrize("kind", ["point", "bar"])
    def test_missing_value_channel_refuses(self, cat_df, kind):
        with pytest.raises(ValueError, match="requires both a category and a value channel"):
            fm.catplot(cat_df, x="cat", kind=kind)


# ---------------------------------------------------------------------------
# The other six kinds are untouched
# ---------------------------------------------------------------------------


# SHA-256 of each kind's rendered SVG, captured on the pre-aggregation code
# (`git stash push -- src/ferrum/plots/distribution.py`) with the `cat_df`
# fixture above. These pin the batch-F byte-identity constraint: wiring
# `Summary` into point/bar must leave every other kind's output untouched.
# Regenerate ONLY alongside a deliberate, disclosed change to those kinds.
_UNTOUCHED_KIND_DIGESTS = {
    "strip": "58df116006b52e53befbca7aaf908bc5f7eec1821a9377957ebc3e848b56013d",
    "swarm": "356c82c30e488705bcf86b8f6c21284987b7f272a7be4c5be89bd84b3b5ddee6",
    "box": "397b96d9a1c250f468ec8de85e96ce38c019a93890082fe2f7fa533de88f14c0",
    "violin": "f685cf8ddcd62bfd75cf10f8fbe4e630faa629446df2c9eb50e417580a630bb5",
    "boxen": "b5769400e6d07cfad179678c6b83ff9c9fdc7084ff0efdbd370ea3381363de74",
    "count": "25480897e3c061c194cabf66911eecc2002d7adefb1f61f4c2ac9748acfdb0bf",
}


@pytest.mark.parametrize("kind", sorted(_UNTOUCHED_KIND_DIGESTS))
def test_non_estimating_kinds_are_byte_identical(cat_df, kind):
    chart = (
        fm.catplot(cat_df, x="cat", kind=kind)
        if kind == "count"
        else fm.catplot(cat_df, x="cat", y="val", kind=kind)
    )
    digest = hashlib.sha256(chart.to_svg().encode()).hexdigest()
    assert digest == _UNTOUCHED_KIND_DIGESTS[kind], (
        f"catplot(kind={kind!r}) output moved -- expected byte-identity with the "
        "pre-aggregation baseline"
    )
