"""Regression pin: ``desugar_errorband``'s ribbon/border layers must bind
``color_field`` through :func:`ferrum.marks._desugar_helpers.nominal_color_channel`,
exactly like the eleven diagnostic desugar sites this batch's ``S3 finding 1``
(python-design-reviewer, ``fix/audit-batch-a-appearance``) already fixed.

Root cause: ``composite.py``'s ``desugar_errorband`` was the one call site
``nominal_color_channel`` was NOT swept onto -- ``_enc_ribbon``/``_enc_border``
bound a bare ``color_field`` string onto a **ribbon** mark and two **line**
marks. Left untyped, an Int64 ``color_field`` (e.g. a numeric group/model-id
column -- entirely legal input to the raw ``mark_errorband`` API) infers
Quantitative -> Continuous, which is inert on ribbon/line: the render
pipeline's ``UnsupportedColorScaleOnMark`` warning fires, the per-group
legend is suppressed, and -- unlike the diagnostic desugars, which only lose
the legend -- the ribbon groups collapse into a single merged band because
nothing partitions the rows by color. A Utf8 ``color_field`` happened to
infer Nominal already, which is why this went unnoticed: the defect is
dtype-dependent.

See ``tests/test_diagnostic_class_column_typing.py`` for the sibling sweep
this shares its root cause and idiom with (that file scopes to
``marks/diagnostic/*``; ``desugar_errorband`` lives in ``marks/composite.py``,
a general composite mark, so it gets its own file per the repo's
findings-scoped test-file convention).
"""

from __future__ import annotations

import random
import warnings

import polars as pl

import ferrum


def _df(color_dtype) -> pl.DataFrame:
    """Two-group frame with enough samples per (x, group) for ``ErrorExtent``
    to compute a real (non-degenerate) interval -- a single sample per group
    collapses to a zero-width band that renders as a ``<rect>``, not the
    ``<path>`` shape a real ribbon draws, which would mask the bug this file
    pins (the design reviewer's exact repro: ``svg.count('<path')`` is 1 for
    a merged band, 2 for two distinct per-group bands).
    """
    rng = random.Random(42)
    rows = []
    labels = ("a", "b") if color_dtype is pl.Utf8 else (0, 1)
    for x in range(1, 6):
        for label in labels:
            for _ in range(10):
                rows.append({"x": float(x), "grp": label, "y": float(x) + rng.gauss(0, 0.2)})
    return pl.DataFrame(rows).with_columns(pl.col("grp").cast(color_dtype))


def _render_and_capture_warnings(chart) -> tuple[str, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, caught


def test_errorband_integer_color_field_splits_into_two_bands_no_warning():
    """Int64 ``color_field`` must render two distinct ribbon bands, exactly
    as the equivalent Utf8 column does -- not one merged band under a
    suppressed continuous colorbar.
    """
    df = _df(pl.Int64)
    chart = ferrum.Chart(df).mark_errorband(color_field="grp").encode(x="x", y="y")
    svg, caught = _render_and_capture_warnings(chart)

    assert "<svg" in svg
    assert svg.count("<path") == 2, f"expected 2 ribbon paths (one per group), got svg: {svg!r}"
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not user_warnings, (
        f"unexpected UserWarning(s): {[str(w.message) for w in user_warnings]}"
    )


def test_errorband_integer_color_field_matches_utf8_path_count():
    """An Int64 hue and the dtype-equivalent Utf8 hue must render the same
    number of ribbon paths -- the bug made only the Int64 case collapse.
    """
    int_svg, _ = _render_and_capture_warnings(
        ferrum.Chart(_df(pl.Int64)).mark_errorband(color_field="grp").encode(x="x", y="y")
    )
    str_svg, _ = _render_and_capture_warnings(
        ferrum.Chart(_df(pl.Utf8)).mark_errorband(color_field="grp").encode(x="x", y="y")
    )
    assert int_svg.count("<path") == str_svg.count("<path") == 2


def test_errorband_integer_color_field_borders_carry_color_too():
    """``borders=True``'s two ``line`` layers must also bind Nominal, not a
    bare Int64 string (the second half of finding 1's ``composite.py:561``).
    """
    df = _df(pl.Int64)
    chart = ferrum.Chart(df).mark_errorband(color_field="grp", borders=True).encode(x="x", y="y")
    spec = chart._build_spec()
    import json

    spec_dict = json.loads(spec.to_json())
    line_layers = [lyr for lyr in spec_dict["layers"] if lyr["mark"] == "line"]
    assert len(line_layers) == 2, "expected lower_border + upper_border line layers"
    for lyr in line_layers:
        enc = lyr.get("encoding", {})
        assert enc.get("color", {}).get("type") == "nominal", (
            f"border layer color channel is not Nominal: {enc.get('color')}"
        )


def test_errorband_no_color_field_unaffected():
    """No ``color_field`` (the default) must stay byte-stable: no color
    channel on the ribbon layer at all.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = ferrum.Chart(df).mark_errorband().encode(x="x", y="y")
    spec = chart._build_spec()
    import json

    spec_dict = json.loads(spec.to_json())
    assert "color" not in spec_dict["layers"][0].get("encoding", {})
