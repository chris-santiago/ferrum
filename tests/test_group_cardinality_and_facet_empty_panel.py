"""RED-proof coverage for wave-1 hardening Task 11.

Issues resolved at Rust's render boundary
(``crates/ferrum-core/src/render/{draw.rs,scene_build.rs,scale_resolve/positional.rs,mod.rs}``,
``crates/ferrum-core/src/render/marks/{segment,image,geoshape,rect}.rs``):

- **#183** (degenerate/high-cardinality ordinal grouping): a categorical axis
  squeezed into a plot area too narrow for its category count used to resolve
  into sub-pixel bands silently. It now warns, naming the category count
  (``RenderWarning::GroupCardinalityDegenerate``). The issue's OWN frozen
  repro groups by a CONTINUOUS-typed column, which resolves a continuous (not
  ordinal) x scale -- the ordinal-site warning above can never fire for that
  shape, since no ordinal arm runs. That symptom ("549KB SVG, one rect, 0
  visible marks, silently") is closed instead at its actual site:
  ``mark_rect()``'s band-relative-width geometry has no meaning without an
  ordinal axis, and now names that mismatch via ``RenderWarning::MarkBuildEmpty``
  (#118) rather than silently drawing nothing.
- **#118** (the zero-mark silent-empty policy): ``MarkBuildResult`` gains an
  ``empty_reason``, populated at five proven-live sites (``mark_segment()``
  missing ``x2``/``y2``, ``mark_image()`` with neither a ``url`` channel nor
  Raster-transform data, ``mark_image()`` under ``CoordPolar``/``CoordGeo``
  (neither of its build paths places anything outside Cartesian pixel
  space), ``mark_geoshape()`` with no geometry data at all, ``mark_rect()``
  needing an ordinal axis it didn't get) and surfaced ONLY at
  ``scene_build``'s dispatch call site as ``RenderWarning::MarkBuildEmpty``.
  The exemption table still holds: a facet panel with no rows for its own
  row/col combination (an incomplete factorial design, a common and
  legitimate shape) stays silent from the per-panel mark-build path; a
  layer whose named ``data_source`` transform output is legitimately empty
  (e.g. a boxplot's ``outliers`` layer on data with no IQR outliers) also
  stays silent, since that call site cannot tell "the transform correctly
  found nothing to keep" apart from "the transform degraded."
- **#240-3** (break-suppression rollup): a chart with N annotation elements
  suppressed by a ``BreakAxis`` gap used to emit one warning per suppressed
  element (and, since a desugar like ``annotate_hline`` emits 2 primitive
  elements per user call, 2N per user-level annotation). Both suppression
  mechanisms (``AnnotationSuppressedByBreak`` for a point coordinate landing
  in the gap, ``AnnotationSpansBreakGap`` for an extent crossing one) now
  roll up to ONE warning per axis per mechanism, naming the count. The
  Display text says "annotation element(s)", not "annotation(s)" -- this
  layer counts desugared primitive specs, with no attribution back to which
  user-facing ``annotate_*()`` call produced which element, so counting
  elements (rather than claiming a user-call count it cannot back up) is the
  honest answer.
"""

import warnings

import polars as pl

import ferrum as fm


def test_catplot_boxplot_500_categories_warns_group_cardinality_degenerate():
    """#183 RED proof: pre-change, a 500-category boxplot resolved its x-axis
    into sub-pixel bands with no diagnostic at all; post-change it names the
    category count and the fact that per-category bands may not render
    visibly. Exercised through the real ``fm.catplot(kind="box")`` entry
    point, not a hand-built scene."""
    n = 500
    df = pl.DataFrame(
        {
            "cat": [f"c{i}" for i in range(n)],
            "value": [float(i % 7) for i in range(n)],
        }
    )
    # An explicit, unremarkable chart width -- comfortably narrower than 500
    # categories could ever occupy at >= 1px each -- so the warning proof
    # rests on the REAL per-panel plot area, not on how wide a default chart
    # happens to be.
    chart = fm.catplot(df, x="cat", y="value", kind="box").properties(width=300)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()

    messages = [str(w.message) for w in caught]
    assert any(
        "500 categories" in m and "per category" in m for m in messages
    ), f"expected a GroupCardinalityDegenerate warning naming 500 categories; got {messages}"


def test_facet_empty_panel_after_filter_stays_silent():
    """#118 exemption-table pin: a facet grid whose row/col combination has no
    rows for one or more cells (an incomplete factorial design) is a common,
    legitimate shape -- not a misconfiguration -- and must not gain a NEW
    per-panel warning from scene_build's own facet-filter continue.

    The grid here has 4 cells (2 row values x 2 col values); only the
    (A, X) and (B, Y) diagonal is populated, so (A, Y) and (B, X) render
    empty. A separate, pre-existing grid-planning-level notice already
    reports exactly this ("facet grid has empty cell(s) with no data") --
    that mechanism is untouched by this task and is deliberately allowed
    through; what must NOT appear is a per-panel `RenderWarning::MarkBuildEmpty`
    ("... drew nothing: ...") from the per-panel mark-build path this task
    changed. (Quality-review finding: the prior wording here asserted
    absence of "skipped"/"left empty", neither of which is a substring any
    warning this task can add actually contains -- a regression that made
    the facet-filtered empty panel emit `MarkBuildEmpty` would have passed
    silently. Asserting on "drew nothing" instead makes this pin able to
    fail.)
    """
    df = pl.DataFrame(
        {
            "row_cat": ["A", "A", "B", "B"],
            "col_cat": ["X", "X", "Y", "Y"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .facet(row="row_cat", col="col_cat")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "<svg" in svg
    messages = [str(w.message) for w in caught]
    assert any(
        "empty cell" in m for m in messages
    ), f"expected the pre-existing grid-planning notice to still fire; got {messages}"
    assert not any(
        "drew nothing" in m for m in messages
    ), f"a facet-filtered empty panel must not ALSO warn from the per-panel mark-build path; got {messages}"


def test_catplot_boxplot_continuous_groupby_column_warns_named_instead_of_silent_chrome_only():
    """#183's OWN frozen repro: "Grouping a transform by a continuous column
    produced a 549KB SVG with one rect and 0 visible marks, silently."

    A `:N`-typed high-cardinality Float64 column resolves a CONTINUOUS x
    scale (Rust's `infer_spec_type`/shorthand-type handling does not make
    every composite-mark layer restate the chart-level type, so the
    per-layer merge loses it -- a separate, escalated Python-side finding),
    so `GroupCardinalityDegenerate` (which only fires on an Ordinal/Band/
    Point scale) cannot fire for this shape. The symptom this issue actually
    reports -- silent, chrome-only output -- is closed at its real site
    instead: `mark_rect()`'s IQR-box layer needs an ordinal axis for its
    band-relative width and doesn't get one, and now says so by name
    (`RenderWarning::MarkBuildEmpty`) rather than silently drawing nothing.
    """
    n = 200
    df = pl.DataFrame(
        {
            "cont": [float(i % 50) for i in range(n)],
            "v": [float(i % 7) for i in range(n)],
        }
    )
    chart = fm.catplot(df, x="cont:N", y="v", kind="box")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "<svg" in svg
    messages = [str(w.message) for w in caught]
    assert any(
        "rect" in m and "ordinal" in m for m in messages
    ), f"expected a MarkBuildEmpty warning naming the rect/ordinal mismatch; got {messages}"


def test_mark_segment_missing_x2_y2_warns_named_empty():
    """#118 RED proof: `mark_segment()` bound to only x/y (a presence-legal
    encoding -- nothing refuses it at construction) used to render a blank
    panel with no diagnostic. Every in-repo producer of a segment layer
    binds all four channels, so a `Some(reason)` here can never fire on a
    ferrum-built chart -- only a hand-built spec missing what the mark needs.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_segment().encode(x="x", y="y")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "<svg" in svg
    messages = [str(w.message) for w in caught]
    assert any(
        "segment" in m and "x2" in m for m in messages
    ), f"expected a MarkBuildEmpty warning naming the missing x2 channel; got {messages}"


def test_mark_image_missing_url_warns_named_empty():
    """#118 RED proof: `mark_image()` with no `url` channel and no
    Raster-transform data used to render nothing with no diagnostic."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_image().encode(x="x", y="y")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "<svg" in svg
    messages = [str(w.message) for w in caught]
    assert any(
        "image" in m and "url" in m for m in messages
    ), f"expected a MarkBuildEmpty warning naming the missing url channel; got {messages}"


def test_mark_image_under_polar_coord_warns_named_empty():
    """#118 RED proof (spec-review cycle-2 residual): `mark_image()` places
    both its url-tile and raster paths in Cartesian pixel space, so a
    `CoordPolar` chart used to render nothing with no diagnostic -- the same
    class the mechanism already covers for segment/rect/geoshape. The
    Cartesian control (same data, no coord override) must render and stay
    silent, pinning that the warning is keyed to the coord.
    """
    # `url` values must be `data:<mime>;base64,<payload>` URLs -- anything
    # else is a documented, unrelated per-row skip (not this task's concern).
    data_url = "data:image/png;base64,iVBORw0KGgo="
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "url": [data_url, data_url, data_url]}
    )

    polar_chart = (
        fm.Chart(df)
        .mark_image()
        .encode(x="x", y="y", url="url")
        .coord(fm.CoordPolar())
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polar_svg = polar_chart.to_svg()
    assert "<svg" in polar_svg
    polar_messages = [str(w.message) for w in caught]
    assert any(
        "image" in m and "CoordPolar" in m for m in polar_messages
    ), f"expected a MarkBuildEmpty warning naming CoordPolar; got {polar_messages}"
    assert "<image" not in polar_svg

    cartesian_chart = fm.Chart(df).mark_image().encode(x="x", y="y", url="url")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cartesian_svg = cartesian_chart.to_svg()
    assert "<image" in cartesian_svg, "the Cartesian control must render"
    assert not caught, f"the Cartesian control must stay silent; got {[str(w.message) for w in caught]}"


def test_mark_geoshape_no_geometry_warns_named_empty():
    """#118 RED proof: `mark_geoshape()` on data with no `__geometry__`
    column at all (never coerced through the GeoJSON path) used to render
    nothing with no diagnostic. A batch whose geometry column is present but
    all-null (a legitimate per-panel/facet degenerate case, spec §6.4) must
    stay silent -- untouched here, see `geoshape.rs`'s own regression test.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_geoshape().encode(x="x", y="y")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "<svg" in svg
    messages = [str(w.message) for w in caught]
    assert any(
        "geoshape" in m and "geometry" in m for m in messages
    ), f"expected a MarkBuildEmpty warning naming the missing geometry; got {messages}"


def test_break_axis_n_in_gap_annotations_roll_up_to_one_warning():
    """#240-3 RED proof, the frozen repro exactly: 4 in-gap `annotate_hline`s
    (each desugaring to a `line` + its own label `text` primitive, so 8
    primitive elements) used to emit 8 near-identical warnings; they now roll
    up into exactly ONE warning naming the axis and the count. The message
    says "annotation element(s)", not "annotation(s)" -- honest about what
    is actually counted (spec-review cycle-2 residual): this layer has no
    attribution back to which user-facing `annotate_hline()` call produced
    which element, so it cannot claim a user-call count of 4.
    """
    df = pl.DataFrame(
        {"category": ["A", "B", "C", "D", "E"], "value": [72.0, 68.0, 75.0, 900.0, 81.0]}
    )
    chart = (
        fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
        + fm.BreakAxis(axis="y", gap=(150, 850))
    )
    for y in (300.0, 400.0, 500.0, 600.0):
        chart = chart + fm.annotate_hline(y=y, label=f"line at {y}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()

    messages = [str(w.message) for w in caught]
    matches = [m for m in messages if "y axis" in m and "suppressed" in m]
    assert len(matches) == 1, (
        f"4 in-gap annotations (8 primitive elements) must roll up into exactly "
        f"ONE warning; got {messages}"
    )
    assert "8 annotation element" in matches[0], (
        f"the count must be honestly labeled as elements, not annotations; got {matches[0]!r}"
    )


def test_empty_data_chart_still_warns_on_unrecognized_color_scheme():
    """Quality-review regression proof (finding #1, "the driver of REVISE"):
    Fix 4's placeholder-discard was broader than the problem. It threw away
    EVERY warning the provisional scale resolve produced, not just the
    pixel-range-derived `GroupCardinalityDegenerate` -- so a chart whose data
    is empty after transforms (`build_scene`'s panel loop never resolves a
    panel to re-emit anything) silently lost `UnknownColorScheme` and every
    sibling scale warning (palette overflow, color-range parse failures,
    ...), breaking this whole batch's "never silent" premise for exactly
    that shape of chart.

    This is the reviewer's exact repro: filter every row away, then bind an
    unrecognized `scheme=` on the color channel. The colorbar still renders
    (the provisional pass resolves the continuous color scale regardless of
    row count), so the scheme mismatch is real and must be reported even
    though no panel ever gets real data to place.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(df)
        .transform(fm.transform_filter("datum.x > 100"))
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=fm.Color("v", scale=fm.SequentialScale(scheme="not-a-scheme")),
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "linearGradient" in svg, "the colorbar must still render on empty data"
    messages = [str(w.message) for w in caught]
    assert any(
        "not-a-scheme" in m and "not recognized" in m for m in messages
    ), f"an empty-data chart must still warn on the unrecognized scheme; got {messages}"


def test_healthy_small_chart_renders_with_zero_warnings():
    """Direct pin for the placeholder-suppression behavior itself (Fix 4's
    actual, narrower contract): a 3-category bar chart at a normal width has
    plenty of pixels per category, so it must render with ZERO warnings --
    in particular, no spurious `GroupCardinalityDegenerate` leaking from the
    provisional metadata-only resolve, which always divides its `(0.0, 1.0)`
    placeholder pixel range by the category count regardless of the chart's
    real size. Guards the narrowed `drop_placeholder_only_warnings` filter:
    it must keep dropping exactly this class from the provisional pass, even
    though `test_empty_data_chart_still_warns_on_unrecognized_color_scheme`
    now proves every OTHER class survives that same pass.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y="v")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()

    assert not caught, f"a healthy 3-category chart must render silently; got {[str(w.message) for w in caught]}"


def test_group_cardinality_degenerate_fans_out_per_facet_panel():
    """Accepted, deliberate behavior (coordinator-adjudicated, not a defect):
    `GroupCardinalityDegenerate` fans out one warning per panel on a faceted
    chart, matching the existing `ColorPaletteOverflowed` convention (which
    also fans out per panel, unrelated to this task). This is a DIFFERENT
    shape from #240-3's break-suppression rollup -- that rollup collapses N
    suppressed elements on ONE axis of ONE render into one warning; this is
    N independent panels, each genuinely degenerate on its own, and each
    panel's warning is actionable on its own (a faceted chart's per-panel
    plot area can differ, e.g. via independent facet sizing). No dedup is
    owed here; this test exists so the fan-out is a documented, intentional
    choice rather than an unnoticed gap (as it was before this test existed).
    """
    n = 60
    df = pl.DataFrame(
        {
            "cat": [f"c{i}" for i in range(n)],
            "v": [float(i % 5) for i in range(n)],
            "facet_col": (["A"] * (n // 2)) + (["B"] * (n // 2)),
        }
    )
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y="v")
        .facet(col="facet_col")
        .properties(width=100)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()

    messages = [str(w.message) for w in caught if "categories" in str(w.message)]
    assert len(messages) == 2, (
        f"2 panels must each independently warn (the ColorPaletteOverflowed "
        f"fan-out convention); got {messages}"
    )
    assert all("60 categor" in m for m in messages), messages
