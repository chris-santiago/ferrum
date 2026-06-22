"""Regression tests for T3.3a — mark extent disambiguation (D-EXTENT-1, marks half).

Covers:
- mark_errorbar / mark_errorband: extent → method
- mark_boxplot: extent → whisker_mult

Each mark must:
1. Accept the new canonical kwarg and produce the correct spec.
2. Accept the old extent= alias (with a DeprecationWarning) and produce an
   identical spec to the canonical call.
3. Raise TypeError when both the canonical kwarg and extent= are supplied.

The canonical-kwarg tests are written so they would FAIL against the
pre-rename signature (extent= only), confirming the rename is in place.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def df_grouped():
    return pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture()
def df_lines():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "y": [1.1, 2.2, 3.3, 4.4, 5.5] * 4,
        }
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _spec_json(chart) -> str:
    return chart._build_spec().to_json()


# ===========================================================================
# mark_errorbar
# ===========================================================================


class TestMarkErrorbarMethodKwarg:
    def test_canonical_method_accepted(self, df_grouped):
        """mark_errorbar(method=...) must be accepted without error."""
        chart = fe.Chart(df_grouped).mark_errorbar(method="stderr").encode(x="g", y="v")
        json_str = _spec_json(chart)
        assert "stderr" in json_str

    def test_canonical_method_default_ci(self, df_grouped):
        """Default method is 'ci'; spec must contain 'ci'."""
        chart = fe.Chart(df_grouped).mark_errorbar().encode(x="g", y="v")
        json_str = _spec_json(chart)
        assert "ci" in json_str

    def test_alias_extent_accepted_with_warning(self, df_grouped):
        """extent= alias is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fe.Chart(df_grouped).mark_errorbar(extent="stdev").encode(x="g", y="v")
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings, "Expected a DeprecationWarning for extent= alias"
        assert "stdev" in _spec_json(chart)

    def test_alias_produces_identical_spec(self, df_grouped):
        """mark_errorbar(extent='stdev') and mark_errorbar(method='stdev') produce identical specs."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            json_alias = _spec_json(
                fe.Chart(df_grouped).mark_errorbar(extent="stdev").encode(x="g", y="v")
            )
        json_canonical = _spec_json(
            fe.Chart(df_grouped).mark_errorbar(method="stdev").encode(x="g", y="v")
        )
        assert json_alias == json_canonical

    def test_both_canonical_and_alias_raises(self, df_grouped):
        """Supplying both method= and extent= must raise TypeError."""
        with pytest.raises(TypeError, match="extent"):
            fe.Chart(df_grouped).mark_errorbar(method="stdev", extent="iqr").encode(x="g", y="v")

    def test_canonical_name_would_fail_on_old_signature(self, df_grouped):
        """This test targets the renamed kwarg 'method=' directly.

        Pre-rename, 'method=' was not a parameter of mark_errorbar(); the
        call would have raised TypeError (unexpected keyword argument).  Now
        it must succeed.
        """
        # This line must not raise.
        chart = fe.Chart(df_grouped).mark_errorbar(method="iqr").encode(x="g", y="v")
        assert "iqr" in _spec_json(chart)

    def test_canonical_at_default_value_and_alias_raises(self, df_grouped):
        """mark_errorbar(method='ci', extent='stdev') must raise TypeError.

        Previously the value-comparison ``method != 'ci'`` misdetected this as
        'canonical not supplied' (because method IS 'ci') and silently accepted
        the alias, discarding the explicit argument.  The sentinel-based
        detection correctly identifies the canonical as explicitly supplied
        (even when it equals the default value) and raises TypeError.
        """
        with pytest.raises(TypeError, match="extent"):
            fe.Chart(df_grouped).mark_errorbar(method="ci", extent="stdev").encode(x="g", y="v")


# ===========================================================================
# mark_errorband
# ===========================================================================


class TestMarkErrorbandMethodKwarg:
    def test_canonical_method_accepted(self, df_lines):
        """mark_errorband(method=...) must be accepted without error."""
        chart = fe.Chart(df_lines).mark_errorband(method="stdev").encode(x="x", y="y")
        json_str = _spec_json(chart)
        assert "stdev" in json_str

    def test_canonical_method_default_ci(self, df_lines):
        """Default method is 'ci'; spec must contain 'ci'."""
        chart = fe.Chart(df_lines).mark_errorband().encode(x="x", y="y")
        assert "ci" in _spec_json(chart)

    def test_alias_extent_accepted_with_warning(self, df_lines):
        """extent= alias is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fe.Chart(df_lines).mark_errorband(extent="stderr").encode(x="x", y="y")
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings, "Expected a DeprecationWarning for extent= alias"
        assert "stderr" in _spec_json(chart)

    def test_alias_produces_identical_spec(self, df_lines):
        """mark_errorband(extent='stderr') and mark_errorband(method='stderr') produce identical specs."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            json_alias = _spec_json(
                fe.Chart(df_lines).mark_errorband(extent="stderr").encode(x="x", y="y")
            )
        json_canonical = _spec_json(
            fe.Chart(df_lines).mark_errorband(method="stderr").encode(x="x", y="y")
        )
        assert json_alias == json_canonical

    def test_both_canonical_and_alias_raises(self, df_lines):
        """Supplying both method= and extent= must raise TypeError."""
        with pytest.raises(TypeError, match="extent"):
            fe.Chart(df_lines).mark_errorband(method="stdev", extent="iqr").encode(x="x", y="y")

    def test_canonical_name_would_fail_on_old_signature(self, df_lines):
        """'method=' must be accepted (was unrecognised before the rename)."""
        chart = fe.Chart(df_lines).mark_errorband(method="iqr").encode(x="x", y="y")
        assert "iqr" in _spec_json(chart)

    def test_canonical_at_default_value_and_alias_raises(self, df_lines):
        """mark_errorband(method='ci', extent='stderr') must raise TypeError.

        Previously the value-comparison ``method != 'ci'`` misdetected this as
        'canonical not supplied' (because method IS 'ci') and silently accepted
        the alias, discarding the explicit argument.  The sentinel-based
        detection correctly identifies the canonical as explicitly supplied
        (even when it equals the default value) and raises TypeError.
        """
        with pytest.raises(TypeError, match="extent"):
            fe.Chart(df_lines).mark_errorband(method="ci", extent="stderr").encode(x="x", y="y")


# ===========================================================================
# mark_boxplot
# ===========================================================================


class TestMarkBoxplotWhiskerMultKwarg:
    def test_canonical_whisker_mult_accepted(self, df_grouped):
        """mark_boxplot(whisker_mult=...) must be accepted without error."""
        chart = fe.Chart(df_grouped).mark_boxplot(whisker_mult=2.0).encode(x="g", y="v")
        # Just confirm it builds; whisker_mult flows into BoxStats.whisker_extent.
        json_str = _spec_json(chart)
        assert "box_stats" in json_str or "whisker" in json_str

    def test_canonical_whisker_mult_min_max(self, df_grouped):
        """whisker_mult='min-max' flows into the spec."""
        chart = fe.Chart(df_grouped).mark_boxplot(whisker_mult="min-max").encode(x="g", y="v")
        assert "min-max" in _spec_json(chart)

    def test_alias_extent_accepted_with_warning(self, df_grouped):
        """extent= alias is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = fe.Chart(df_grouped).mark_boxplot(extent="min-max").encode(x="g", y="v")
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings, "Expected a DeprecationWarning for extent= alias"
        assert "min-max" in _spec_json(chart)

    def test_alias_produces_identical_spec(self, df_grouped):
        """mark_boxplot(extent='min-max') and mark_boxplot(whisker_mult='min-max') produce identical specs."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            json_alias = _spec_json(
                fe.Chart(df_grouped).mark_boxplot(extent="min-max").encode(x="g", y="v")
            )
        json_canonical = _spec_json(
            fe.Chart(df_grouped).mark_boxplot(whisker_mult="min-max").encode(x="g", y="v")
        )
        assert json_alias == json_canonical

    def test_both_canonical_and_alias_raises(self, df_grouped):
        """Supplying both whisker_mult= and extent= must raise TypeError."""
        with pytest.raises(TypeError, match="extent"):
            fe.Chart(df_grouped).mark_boxplot(whisker_mult=2.0, extent=1.5).encode(x="g", y="v")

    def test_canonical_name_would_fail_on_old_signature(self, df_grouped):
        """'whisker_mult=' must be accepted (was unrecognised before the rename)."""
        chart = fe.Chart(df_grouped).mark_boxplot(whisker_mult=2.0).encode(x="g", y="v")
        json_str = _spec_json(chart)
        # 2.0 flows into BoxStats.whisker_extent; box layers must still be present.
        assert "lower_whisker" in json_str

    def test_default_whisker_mult_is_1_5(self, df_grouped):
        """Default whisker_mult=1.5 must produce 'min-max'-free spec."""
        chart = fe.Chart(df_grouped).mark_boxplot().encode(x="g", y="v")
        json_str = _spec_json(chart)
        assert "min-max" not in json_str

    def test_canonical_at_default_value_and_alias_raises(self, df_grouped):
        """mark_boxplot(whisker_mult=1.5, extent='min-max') must raise TypeError.

        Previously the value-comparison `whisker_mult != 1.5` misdetected this
        as "canonical not supplied" and silently accepted the alias.  The
        sentinel-based detection correctly identifies the canonical as explicitly
        supplied (even at its default value) and raises TypeError.
        """
        with pytest.raises(TypeError, match="extent"):
            fe.Chart(df_grouped).mark_boxplot(whisker_mult=1.5, extent="min-max").encode(
                x="g", y="v"
            )

    def test_whisker_mult_str_value_accepted(self, df_grouped):
        """mark_boxplot(whisker_mult='min-max') must be accepted as a str value.

        Previously the annotation narrowed to `float`, so a `str` value was
        rejected at the type-check level and the downstream `_extent_to_box`
        str-branch was unreachable via the public API.
        """
        chart = fe.Chart(df_grouped).mark_boxplot(whisker_mult="min-max").encode(x="g", y="v")
        assert "min-max" in _spec_json(chart)


# ===========================================================================
# shared_extent is preserved (data-domain sense, mark_violin/boxen)
# ===========================================================================


def test_shared_extent_untouched():
    """heavy_stat.shared_extent (violin/boxen) must not have been renamed."""
    import inspect

    from ferrum.marks.heavy_stat import desugar_violin

    sig = inspect.signature(desugar_violin)
    assert "shared_extent" in sig.parameters, (
        "desugar_violin.shared_extent must NOT have been renamed — "
        "it is the data-domain extent sense preserved by D-EXTENT-1"
    )
