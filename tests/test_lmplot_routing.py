"""Batch-D T13 (spec §4.7/M6c): `lmplot`/`regplot`/`residplot` routing.

Two carried obligations close here:

1. **The T9-found `y="fitted"` bug.** `lmplot`'s `logistic`/`glm`/`robust`
   branches used to build ``.transform(<fit>).mark_line().encode(x=x,
   y="fitted")`` against a transform whose real fitted-curve output schema
   is ``x, y, ci_lower, ci_upper`` (`FIT_OUTPUT_COLUMNS`, shared with
   `Smooth`) -- there has never been a `"fitted"` column. **RED, reproduced
   directly against the pre-fix tree** (not a hand-built shape): before this
   fix, ``fm.lmplot(df, x="x", y="y", method="logistic",
   scatter=False).to_svg()`` raised
   ``ValueError: unknown column 'fitted' referenced by an encoding``. Every
   test below that renders a `logistic`/`glm`/`robust` chart pins the fix by
   construction -- none of them could pass against the pre-fix branches.
2. **The `_VALID_METHODS` widening** (commit d787dbc0). Resolved by
   *routing*, not a dated-note carve-out: `regression.py`'s
   `_SMOOTH_ROUTED_METHODS` is now derived from
   `transform_registry("smooth")["methods"]`, so `"quadratic"`/`"cubic"`/
   `"log"`/`"sqrt"`/`"gam"` are real `lmplot(method=...)` values, each
   rendering through `mark_smooth`. `tests/test_lmplot_valid_methods_
   registry_pin.py` pins the derivation itself; this module pins that the
   newly-accepted methods actually render.

Scoped per the Python test-file convention (`tests/marks/test_boxen_palette.py`'s
module docstring): net-new figure-function routing behavior gets its own
module rather than growing `test_smooth_methods.py` (which scopes to
`Smooth`/`mark_smooth` itself). `tests/test_lmplot_residplot_order.py` covers
`lmplot`/`residplot`'s bare `order=` forwarding (`method="lm"`'s own default,
net-new as of T9); the `order=2 == method="quadratic"` equivalence pin lives
in `tests/test_smooth_methods.py::test_lmplot_order_2_matches_mark_smooth_
quadratic` (written by T9). This module additionally covers the T13 round-2
`order=`-as-`degree=` conflict-forwarding for `method="quadratic"`/`"cubic"`
(a distinct, newer forwarding path -- see the "order= forwards..." section
below), which belongs here rather than either of those two modules since it
is `lmplot`'s own routing decision, not `Smooth`'s.

`order=` forwards as `degree=` under ONE rule for every
`_SMOOTH_ROUTED_METHODS` member (commit d787dbc0): log/sqrt/gam raise the same
typed refusal `mark_smooth` gives directly, and loess picks up its own
local-fit `degree`. `order=None` (the default) is the only "no explicit
degree" value; an explicit `order=1` forwards like any other value, so
loess's local-fit degree=1 (a real, different fit from its degree=2 default)
is reachable through `lmplot` (see
`test_loess_order_1_is_a_real_linear_local_fit_not_a_noop`,
`test_order_conflicting_with_fixed_degree_raises`,
`test_order_forwarded_and_refused_for_no_degree_methods`); a bare call is the
only shape that forwards nothing
(`test_widened_smooth_method_bare_call_does_not_forward_order`). A negative
`order` raises a typed `ValueError`, not an untyped PyO3 `OverflowError`
(`test_negative_order_raises_typed_value_error`).

A dedicated method fits one curve per group of the scatter's color, in that
group's color, whatever door colors it:
`test_dedicated_method_hue_fits_each_group_in_its_color` pins `hue=`, and
`test_dedicated_method_color_channel_fits_each_group_in_its_color` is one
matrix over every channel `ferrum.encoding._aliases._ENCODING_ALIASES`
resolves onto `color` (`color` directly, `fill`/`stroke` via the alias table)
x both doors (bare keyword, `encode={...}` dict) x all three dedicated
methods, so a new alias door cannot slip past it.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.plots.regression import _SMOOTH_ROUTED_METHODS

_DEDICATED_FIT_METHODS = ("logistic", "glm", "robust")


@pytest.fixture
def binary_xy_df() -> pl.DataFrame:
    """Deterministic, non-degenerate binary-target data for `logistic`: a
    real sigmoid separation (not all-0/all-1, not a single unique x -- see
    `tests/test_transform_edge_cases.py::TestLogistic` for those refusal
    cases, which this fixture deliberately avoids)."""
    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(-3.0, 3.0, n)
    p = 1.0 / (1.0 + np.exp(-x))
    y = (rng.uniform(size=n) < p).astype(float)
    return pl.DataFrame({"x": x, "y": y})


def _polyline_points(svg: str) -> list[tuple[float, float]]:
    """Extract the (x, y) coordinate pairs from the first `<polyline>` in an
    SVG (the fit line -- `mark_line` always renders as a polyline, never a
    `<path>`, matching `test_smooth_methods.py`'s own convention for this
    same assertion shape)."""
    m = re.search(r'<polyline[^>]*points="([^"]+)"', svg)
    assert m is not None, "expected a <polyline> element (the fit line)"
    pts = []
    for pair in m.group(1).strip().split():
        sx, sy = pair.split(",")
        pts.append((float(sx), float(sy)))
    return pts


def _polyline_strokes(svg: str) -> set[str]:
    """Every `<polyline stroke="...">` color in an SVG -- used to check the
    fit line's rendered stroke against the scatter's hue palette."""
    return set(re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg))


def _circle_fills(svg: str) -> set[str]:
    """Every `<circle fill="...">` color in an SVG -- `mark_point` always
    renders as `<circle>`, so this is the scatter's per-hue-level palette."""
    return set(re.findall(r'<circle[^>]*fill="([^"]+)"', svg))


# ---------------------------------------------------------------------------
# The probe's logistic repro (F-L10-01): scatter=False renders a real fit line
# ---------------------------------------------------------------------------


def test_logistic_scatter_false_renders_fit_line(binary_xy_df):
    svg = fm.lmplot(binary_xy_df, x="x", y="y", method="logistic", scatter=False).to_svg()
    assert svg.count("<polyline") == 1
    assert "NaN" not in svg


def test_logistic_fit_line_is_sigmoid_shaped_not_raw_scatter(binary_xy_df):
    """Discriminates a real logistic fit from the raw scatter that a
    field-resolution bug could accidentally render instead: a sigmoid fit
    line is monotone in x and its y (in SVG screen space) stays within a
    bounded band -- the raw (x, y) scatter of binary 0/1 targets, plotted as
    a connected polyline over unsorted rows, is neither."""
    svg = fm.lmplot(binary_xy_df, x="x", y="y", method="logistic", scatter=False).to_svg()
    pts = _polyline_points(svg)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert xs == sorted(xs), "fit line must be monotone in x (evaluated over a sorted grid)"
    # Screen-space y is monotone too (a sigmoid is monotone in y for this
    # fixture's positive slope); either direction is fine, just not flat and
    # not oscillating like a scatter of raw jittered points would.
    y_diffs = [b - a for a, b in zip(ys, ys[1:])]
    assert all(d <= 0 for d in y_diffs) or all(d >= 0 for d in y_diffs)


def test_logistic_scatter_true_default_renders():
    """scatter=True (the default) still works with the fixed routing --
    exercises `_merge_layers` against the new directly-built `_layers` fit
    chart (not just the scatter=False/no-merge path above)."""
    rng = np.random.default_rng(1)
    n = 150
    x = np.linspace(-3.0, 3.0, n)
    p = 1.0 / (1.0 + np.exp(-x))
    y = (rng.uniform(size=n) < p).astype(float)
    df = pl.DataFrame({"x": x, "y": y})
    svg = fm.lmplot(df, x="x", y="y", method="logistic").to_svg()
    assert svg.count("<polyline") == 1
    assert "NaN" not in svg


# ---------------------------------------------------------------------------
# glm / robust: fit line + CI band render; ci=None suppresses the band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_dedicated_method_renders_fit_line_and_ci_band(binary_xy_df, curved_xy_df, method):
    # py-quality round-1 (S2): logistic was previously pinned only negatively
    # (`ci=None` -> 0 paths, below); this positive pin (`ci=95` -> 1 path)
    # was available for free -- logistic does render a band.
    df = binary_xy_df if method == "logistic" else curved_xy_df
    svg = fm.lmplot(df, x="x", y="y", method=method, scatter=False).to_svg()
    assert svg.count("<polyline") == 1, "exactly one fit-line polyline"
    assert svg.count("<path") == 1, "exactly one CI-ribbon path"
    assert "NaN" not in svg


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_dedicated_method_ci_none_suppresses_band(curved_xy_df, method):
    df = curved_xy_df
    if method == "logistic":
        # logistic needs a binary y; curved_xy_df's y is continuous.
        rng = np.random.default_rng(2)
        n = 150
        x = np.linspace(-3.0, 3.0, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = (rng.uniform(size=n) < p).astype(float)
        df = pl.DataFrame({"x": x, "y": y})
    svg = fm.lmplot(df, x="x", y="y", method=method, ci=None, scatter=False).to_svg()
    assert svg.count("<polyline") == 1
    assert svg.count("<path") == 0, "ci=None must suppress the ribbon layer entirely"


# ---------------------------------------------------------------------------
# scatter=False works for every dedicated fit-family method (T9's carried note)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_scatter_false_works(binary_xy_df, curved_xy_df, method):
    df = binary_xy_df if method == "logistic" else curved_xy_df
    chart = fm.lmplot(df, x="x", y="y", method=method, scatter=False)
    svg = chart.to_svg()
    assert svg.startswith("<svg")


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_regplot_routes_dedicated_methods_through_lmplot(binary_xy_df, curved_xy_df, method):
    """`regplot` forwards every parameter unchanged to `lmplot` -- one smoke
    test per dedicated method confirms the delegation still works post-fix."""
    df = binary_xy_df if method == "logistic" else curved_xy_df
    svg = fm.regplot(df, x="x", y="y", method=method, scatter=False).to_svg()
    assert svg.startswith("<svg")
    assert "NaN" not in svg


# ---------------------------------------------------------------------------
# hue= on a dedicated method: one fit per group, each in its group's scatter
# color. The assertion reads each fit's stroke color, since a polyline count
# alone passes whether the fits are colored honestly or not.
# ---------------------------------------------------------------------------


def _grouped(df: pl.DataFrame) -> pl.DataFrame:
    """*df* with a two-level group ``g`` alternating over its rows."""
    return df.with_columns(pl.Series("g", ["a" if i % 2 == 0 else "b" for i in range(df.height)]))


def _assert_one_fit_per_group_in_its_color(svg: str) -> None:
    assert "NaN" not in svg
    assert svg.count("<polyline") == 2, "one fit line per group"
    scatter_fills = _circle_fills(svg)
    assert len(scatter_fills) == 2, f"expected 2 group colors on the scatter, got {scatter_fills}"
    assert _polyline_strokes(svg) == scatter_fills, "each fit draws its group's scatter color"


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_dedicated_method_hue_fits_each_group_in_its_color(binary_xy_df, curved_xy_df, method):
    """`logistic`/`glm`/`robust` fit one curve per hue level, each drawn in
    that level's scatter color, as the `mark_smooth`-routed methods do.

    Rejects one pooled fit over every group (one polyline) and a fit drawn
    in a color no group has."""
    df = _grouped(binary_xy_df if method == "logistic" else curved_xy_df)
    chart = fm.lmplot(df, x="x", y="y", method=method, hue="g")
    _assert_one_fit_per_group_in_its_color(chart.to_svg())
    assert '"groupby":["g"]' in chart.to_spec().to_json().replace(" ", "")


def test_dedicated_method_hue_absent_keeps_default_accent_stroke(curved_xy_df):
    """Without `hue=`, there is no palette to collide with -- the fit line
    keeps the ordinary theme accent stroke, matching `lm`/`loess`'s own
    no-hue default (no visual regression for the non-hue case)."""
    svg = fm.lmplot(curved_xy_df, x="x", y="y", method="robust", scatter=False).to_svg()
    assert _polyline_strokes(svg) == {"#2563eb"}


# ---------------------------------------------------------------------------
# color= / fill= / stroke=, each via a keyword or an encode={...} dict, on a
# dedicated method: every channel `ferrum.encoding._aliases._ENCODING_ALIASES`
# resolves onto `color` groups the fit as `hue=` does. The population is
# derived from the alias table, not hand-listed keys, so an alias door
# (`fill=`, `stroke=`) cannot be missed. `hue=` keeps its own test above
# since it has no `encode=` counterpart (`hue` is not a `Chart.encode()`
# channel).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
@pytest.mark.parametrize("form", ["keyword", "encode"])
@pytest.mark.parametrize("channel", ["color", "fill", "stroke"])
def test_dedicated_method_color_channel_fits_each_group_in_its_color(
    binary_xy_df, curved_xy_df, channel, form, method
):
    """Same contract as `test_dedicated_method_hue_fits_each_group_in_its_color`
    for every (channel, form) door the alias policy resolves onto `color`:
    a bare keyword (merged into the scatter's encoding) and
    `encode={channel: ...}` (applied to the whole figure) both group the
    fit, so neither door leaves one pooled curve beside a colored scatter."""
    df = _grouped(binary_xy_df if method == "logistic" else curved_xy_df)
    kwargs = {"encode": {channel: "g"}} if form == "encode" else {channel: "g"}
    _assert_one_fit_per_group_in_its_color(
        fm.lmplot(df, x="x", y="y", method=method, **kwargs).to_svg()
    )


# ---------------------------------------------------------------------------
# T12 closure: the widened registry smooth methods route through lmplot
# ---------------------------------------------------------------------------


def test_widened_methods_are_exactly_the_new_registry_smooth_names():
    assert _SMOOTH_ROUTED_METHODS - {"lm", "loess"} == {
        "quadratic",
        "cubic",
        "log",
        "sqrt",
        "gam",
    }


@pytest.mark.parametrize("method", ["quadratic", "cubic", "log", "sqrt", "gam"])
def test_widened_smooth_method_renders_via_lmplot(curved_xy_df, method):
    svg = fm.lmplot(curved_xy_df, x="x", y="y", method=method, scatter=False).to_svg()
    assert svg.count("<polyline") == 1
    assert "NaN" not in svg


@pytest.mark.parametrize("method", ["quadratic", "cubic", "log", "sqrt", "gam"])
def test_widened_smooth_method_matches_direct_mark_smooth(curved_xy_df, method):
    """`lmplot(method=<widened>)` must render byte-identically to calling
    `mark_smooth(method=<widened>)` directly -- pins that the new branch
    forwards nothing beyond `method`/`ci`/`x_range`."""
    lmplot_svg = fm.lmplot(
        curved_xy_df, x="x", y="y", method=method, ci=None, scatter=False, show_metrics=False
    ).to_svg()
    direct_svg = (
        fm.Chart(curved_xy_df).mark_smooth(method=method, ci=None).encode(x="x", y="y").to_svg()
    )
    assert lmplot_svg == direct_svg


@pytest.mark.parametrize("method", ["quadratic", "cubic", "log", "sqrt", "gam"])
def test_widened_smooth_method_bare_call_does_not_forward_order(curved_xy_df, method):
    """A bare call (`order=` omitted entirely -- `lmplot`'s `None` default)
    must NOT forward any degree to these methods -- `quadratic`/`cubic`
    would refuse a conflicting degree and `log`/`sqrt`/`gam` refuse any
    `degree=` at all (Smooth's own degree-acceptance matrix). Batch-D T13
    round 4 moved the "no explicit degree" sentinel from `order == 1` to
    `order is None` (round-3's docstring falsely claimed `order=1` was a
    harmless no-op for `loess`, when it never forwarded at all -- see
    `_smooth_degree_kwargs`'s docstring "History" note), so a bare call is
    now the ONLY shape that doesn't forward -- an explicit `order=1` forwards
    like any other value (see `test_order_conflicting_with_fixed_degree_raises`
    and `test_order_forwarded_and_refused_for_no_degree_methods`, both of
    which now include `order=1`)."""
    svg = fm.lmplot(curved_xy_df, x="x", y="y", method=method, scatter=False).to_svg()
    assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# order= forwards as degree= UNIFORMLY for every `_SMOOTH_ROUTED_METHODS`
# member (commit d787dbc0). The "no explicit degree" sentinel is
# `order is None`, so an EXPLICIT `order=1` forwards like any other value.
# A bare call (`order` omitted) is
# the only shape that still doesn't forward -- pinned above by
# `test_widened_smooth_method_bare_call_does_not_forward_order`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "matching_order"), [("quadratic", 2), ("cubic", 3)])
def test_order_matching_fixed_degree_is_accepted_as_noop(curved_xy_df, method, matching_order):
    with_order = fm.lmplot(
        curved_xy_df,
        x="x",
        y="y",
        method=method,
        order=matching_order,
        scatter=False,
        show_metrics=False,
    ).to_svg()
    bare = fm.lmplot(
        curved_xy_df, x="x", y="y", method=method, scatter=False, show_metrics=False
    ).to_svg()
    assert with_order == bare


@pytest.mark.parametrize(
    ("method", "conflicting_order"),
    [("quadratic", 1), ("quadratic", 3), ("cubic", 1), ("cubic", 2)],
)
def test_order_conflicting_with_fixed_degree_raises(curved_xy_df, method, conflicting_order):
    """`order=1` is included alongside the pre-existing conflicting values --
    batch-D T13 round 4: `1` is no longer the "no explicit degree" sentinel,
    so an EXPLICIT `order=1` for `quadratic`/`cubic` (fixed degree `2`/`3`)
    now conflicts exactly like any other non-matching value."""
    with pytest.raises(ValueError, match="conflicts with method"):
        fm.lmplot(
            curved_xy_df, x="x", y="y", method=method, order=conflicting_order, scatter=False
        ).to_svg()


@pytest.mark.parametrize("method", ["log", "sqrt", "gam"])
@pytest.mark.parametrize("order", [1, 2])
def test_order_forwarded_and_refused_for_no_degree_methods(curved_xy_df, method, order):
    """`log`/`sqrt`/`gam` have no degree concept at all (commit d787dbc0):
    `order=` forwards as `degree=` UNIFORMLY for every
    `_SMOOTH_ROUTED_METHODS` member (one rule, not a per-method carve-out),
    so an explicit `order=` for these three now reaches `Smooth`'s own
    typed refusal -- the same error a direct `mark_smooth(method=...,
    degree=...)` call raises -- rather than being silently dropped.
    `order=1` is included (`1` is not the "no explicit
    degree" sentinel -- see `test_widened_smooth_method_bare_call_does_not_
    forward_order` for the shape that IS still a no-forward)."""
    with pytest.raises(ValueError, match="degree is not valid for method"):
        fm.lmplot(curved_xy_df, x="x", y="y", method=method, order=order, scatter=False).to_svg()


def test_loess_order_2_matches_default_local_fit_degree(curved_xy_df):
    """`loess`'s own `degree` parameter is the LOCAL-FIT degree (1 or 2), not
    a polynomial order -- forwarding `order=2` under the unified rule lands
    on loess's own default local-fit degree (2), so it must be a byte-
    identical no-op against the bare call, not a behavior change."""
    with_order = fm.lmplot(
        curved_xy_df, x="x", y="y", method="loess", order=2, scatter=False, show_metrics=False
    ).to_svg()
    bare = fm.lmplot(
        curved_xy_df, x="x", y="y", method="loess", scatter=False, show_metrics=False
    ).to_svg()
    assert with_order == bare


def test_loess_order_1_is_a_real_linear_local_fit_not_a_noop(curved_xy_df):
    """`order=1` is not the "no explicit degree" sentinel (that is
    `order=None`, commit d787dbc0), so it reaches loess's own
    local-fit degree matrix as a REAL, distinguishable value -- a
    genuinely different (linear) local fit from the bare call's default
    (loess's own default local-fit degree, `2`)."""
    bare = fm.lmplot(
        curved_xy_df, x="x", y="y", method="loess", ci=None, scatter=False, show_metrics=False
    ).to_svg()
    order_1 = fm.lmplot(
        curved_xy_df,
        x="x",
        y="y",
        method="loess",
        order=1,
        ci=None,
        scatter=False,
        show_metrics=False,
    ).to_svg()
    direct_degree_1 = (
        fm.Chart(curved_xy_df)
        .mark_smooth(method="loess", ci=None, degree=1)
        .encode(x="x", y="y")
        .to_svg()
    )
    assert order_1 != bare, "order=1 must be a real forward, not a no-op against the default"
    assert order_1 == direct_degree_1


def test_loess_order_outside_local_fit_range_raises(curved_xy_df):
    """`order=3` forwarded as `degree=3` to `loess` is outside the valid
    local-fit degree range ({1, 2}) -- reaches the same typed refusal a
    direct `mark_smooth(method="loess", degree=3)` call raises."""
    with pytest.raises(ValueError, match="LOESS degree must be 1 or 2"):
        fm.lmplot(curved_xy_df, x="x", y="y", method="loess", order=3, scatter=False).to_svg()


@pytest.mark.parametrize("method", ["quadratic", "cubic", "log", "sqrt", "gam"])
def test_widened_smooth_method_hue_groupby_renders(curved_xy_df, method):
    df = curved_xy_df.with_columns(
        pl.Series("g", ["a" if i % 2 == 0 else "b" for i in range(curved_xy_df.height)])
    )
    svg = fm.lmplot(df, x="x", y="y", method=method, hue="g", scatter=False).to_svg()
    assert svg.startswith("<svg")
    assert "NaN" not in svg


# ---------------------------------------------------------------------------
# order=-1: the Rust `degree` parameter is unsigned, so a negative Python int
# used to fail the PyO3 conversion with an untyped `OverflowError` rather
# than the typed `ValueError` `lmplot`'s `Raises` section promises (batch-D
# T13 round-4 spec-review ⚠️). `_smooth_degree_kwargs` now refuses it
# directly, before it ever reaches the Rust boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["lm", "loess", "quadratic", "cubic", "log", "sqrt", "gam"])
def test_negative_order_raises_typed_value_error(curved_xy_df, method):
    with pytest.raises(ValueError, match="non-negative integer"):
        fm.lmplot(curved_xy_df, x="x", y="y", method=method, order=-1, scatter=False).to_svg()


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_negative_order_ignored_for_dedicated_methods(binary_xy_df, curved_xy_df, method):
    """`order` has no meaning at all for `logistic`/`glm`/`robust` (a
    different fit family, no `order`/`degree` parameter) -- these methods
    never call `_smooth_degree_kwargs`, so no validation, typed or
    otherwise, applies to `order` for them."""
    df = binary_xy_df if method == "logistic" else curved_xy_df
    svg = fm.lmplot(df, x="x", y="y", method=method, order=-1, scatter=False).to_svg()
    assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# logx= still applies for the dedicated fit-family methods (regression guard:
# these bypass the _PendingMark/desugar scale-propagation path other methods
# use -- see regression.py's `logx` block comment)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _DEDICATED_FIT_METHODS)
def test_dedicated_method_logx_changes_rendered_output(binary_xy_df, curved_xy_df, method):
    df = binary_xy_df if method == "logistic" else curved_xy_df
    with_log = fm.lmplot(df, x="x", y="y", method=method, logx=True, scatter=False).to_svg()
    without_log = fm.lmplot(df, x="x", y="y", method=method, logx=False, scatter=False).to_svg()
    assert with_log != without_log


# ---------------------------------------------------------------------------
# Byte-identity: untouched (lm/loess) code paths are unaffected by the
# widened elif chain -- reconstructing the equivalent chart by hand pins that
# the "lm"/"loess" branches forward exactly what they forwarded before.
# ---------------------------------------------------------------------------


def test_lm_default_path_matches_hand_built_equivalent(curved_xy_df):
    lmplot_svg = fm.lmplot(
        curved_xy_df, x="x", y="y", method="lm", scatter=False, show_metrics=False
    ).to_svg()
    direct_svg = (
        fm.Chart(curved_xy_df)
        .mark_smooth(method="lm", ci=0.95, degree=1)
        .encode(x="x", y="y")
        .to_svg()
    )
    assert lmplot_svg == direct_svg


def test_loess_default_path_matches_hand_built_equivalent(curved_xy_df):
    # show_metrics=False on both sides: T13 round 2 forwards `show_metrics`
    # (Smooth's `inject_metrics` is generic across every method, not just
    # "lm") so a bare `lmplot(method="loess")` now DOES render the corner
    # annotation by default -- mirrors `test_lm_default_path_matches_
    # hand_built_equivalent`'s existing show_metrics=False convention.
    lmplot_svg = fm.lmplot(
        curved_xy_df, x="x", y="y", method="loess", scatter=False, show_metrics=False
    ).to_svg()
    direct_svg = (
        fm.Chart(curved_xy_df).mark_smooth(method="loess", ci=0.95).encode(x="x", y="y").to_svg()
    )
    assert lmplot_svg == direct_svg


def test_loess_show_metrics_now_renders_corner_annotation(curved_xy_df):
    """Discriminates the show_metrics forwarding fix (batch-D T13 round 2,
    spec-review ⚠️ / py-quality non-blocking finding): before this fix,
    `show_metrics=True` (lmplot's default) was silently dropped for every
    non-"lm" method -- a bare `lmplot(method="loess")` rendered no corner
    annotation regardless of `show_metrics=`. `Smooth`'s `inject_metrics` is
    a generic transform kwarg (verified against `smooth.rs`: every
    `SmoothMethod` arm accepts it), so it now applies to "loess" too."""
    svg = fm.lmplot(curved_xy_df, x="x", y="y", method="loess", scatter=False).to_svg()
    assert "R²" in svg, "expected the R²/RMSE/MAE corner annotation text"
    svg_off = fm.lmplot(
        curved_xy_df, x="x", y="y", method="loess", scatter=False, show_metrics=False
    ).to_svg()
    assert "R²" not in svg_off


def test_robust_show_metrics_renders_corner_annotation(curved_xy_df):
    """`Robust` also exposes `inject_metrics` (unlike `Logistic`/`Glm`), so
    `method="robust"` gets the same corner annotation as the smooth-routed
    methods -- batch-D T13 round-2 finding 3."""
    svg = fm.lmplot(curved_xy_df, x="x", y="y", method="robust", scatter=False).to_svg()
    assert "R²" in svg
    svg_off = fm.lmplot(
        curved_xy_df, x="x", y="y", method="robust", scatter=False, show_metrics=False
    ).to_svg()
    assert "R²" not in svg_off


@pytest.mark.parametrize("method", ["logistic", "glm"])
def test_dedicated_method_show_metrics_silently_has_no_effect(binary_xy_df, curved_xy_df, method):
    """`Logistic`/`Glm` expose no `inject_metrics` param at all -- verified
    against `transform_registry`. `show_metrics=True` (the default) and
    `show_metrics=False` must render identically for these two, matching the
    docstring's disclosed scope (not a silent drop of a real capability)."""
    df = binary_xy_df if method == "logistic" else curved_xy_df
    on = fm.lmplot(df, x="x", y="y", method=method, scatter=False).to_svg()
    off = fm.lmplot(df, x="x", y="y", method=method, scatter=False, show_metrics=False).to_svg()
    assert on == off
    assert "R²" not in on


# ---------------------------------------------------------------------------
# truncate=False (batch-D T13 round-2 fix; py-quality finding 3b / spec-review
# ⚠️): forwarded to Robust via its x_range param (Robust accepts it); refused
# for logistic/glm (neither transform accepts x_range at all -- no silent
# clip-anyway).
# ---------------------------------------------------------------------------


def test_truncate_false_forwarded_to_robust_changes_output(curved_xy_df):
    truncated = fm.lmplot(
        curved_xy_df, x="x", y="y", method="robust", scatter=False, truncate=True
    ).to_svg()
    extended = fm.lmplot(
        curved_xy_df, x="x", y="y", method="robust", scatter=False, truncate=False
    ).to_svg()
    assert truncated != extended


@pytest.mark.parametrize("method", ["logistic", "glm"])
def test_truncate_false_refused_for_logistic_and_glm(binary_xy_df, curved_xy_df, method):
    df = binary_xy_df if method == "logistic" else curved_xy_df
    with pytest.raises(ValueError, match="truncate=False"):
        fm.lmplot(df, x="x", y="y", method=method, scatter=False, truncate=False)


def test_truncate_false_still_works_for_smooth_routed_methods(curved_xy_df):
    """Regression guard: the new `truncate=False` refusal is scoped to
    `{"logistic", "glm"}` only -- every mark_smooth-routed method keeps its
    pre-existing `x_range` forwarding untouched."""
    truncated = fm.lmplot(
        curved_xy_df, x="x", y="y", method="lm", scatter=False, truncate=True
    ).to_svg()
    extended = fm.lmplot(
        curved_xy_df, x="x", y="y", method="lm", scatter=False, truncate=False
    ).to_svg()
    assert truncated != extended
