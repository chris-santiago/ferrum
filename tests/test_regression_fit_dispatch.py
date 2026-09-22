"""#163 (Python half): `lmplot`'s fit is built by ONE `_build_fit` dispatch.

The dispatch replaced a four-arm `if/elif/.../else: unreachable` chain. These
tests pin its two contracts: every accepted method has a builder (so a new
registry smooth method, or a new dedicated method, cannot silently fall
through), and an unknown method is refused with the exact `ValueError`
`lmplot` itself raises at entry.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.plots import regression
from ferrum.plots.regression import (
    _FIT_BUILDERS,
    _VALID_METHODS,
    _build_fit,
    _build_glm_fit,
    _build_logistic_fit,
    _build_robust_fit,
    _build_smooth_fit,
)


def _request() -> regression._FitRequest:
    return regression._FitRequest(
        data=pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 2.5, 4.0]}),
        x="x",
        y="y",
        x_field="x",
        y_field="y",
        hue=None,
        ci=0.95,
        line_kwargs={},
        x_bins=None,
        x_estimator=None,
        order=None,
        x_range=None,
        metrics=False,
        color_active=False,
        has_scatter=True,
    )


def test_every_accepted_method_has_a_builder() -> None:
    assert set(_FIT_BUILDERS) == _VALID_METHODS


_DEDICATED_BUILDER_BY_METHOD = {
    "logistic": _build_logistic_fit,
    "glm": _build_glm_fit,
    "robust": _build_robust_fit,
}


@pytest.mark.parametrize("method", sorted(_VALID_METHODS))
def test_every_accepted_method_dispatches_to_a_fit_chart(method: str) -> None:
    # Restated independently of `_FIT_BUILDERS`' own construction, so a
    # table entry pointed at the wrong builder (e.g. `"glm":
    # _build_robust_fit`, or `"logistic": _build_glm_fit`) fails here by
    # name instead of merely producing *some* `fm.Chart`.
    expected_builder = _DEDICATED_BUILDER_BY_METHOD.get(method, _build_smooth_fit)
    assert _FIT_BUILDERS[method] is expected_builder

    fit = _build_fit(method, _request())
    assert isinstance(fit, fm.Chart)


def test_unknown_method_refused_with_lmplot_entry_text() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError) as from_lmplot:
        fm.lmplot(df, x="x", y="y", method="bogus")
    with pytest.raises(ValueError, match=re.escape(str(from_lmplot.value))):
        _build_fit("bogus", _request())
    assert "lmplot: method must be one of" in str(from_lmplot.value)


def test_accepted_method_without_builder_fails_by_name(monkeypatch) -> None:
    # A registry method that `_VALID_METHODS` accepts but no builder covers
    # must be named, not reach an unreachable `else`.
    monkeypatch.setattr(regression, "_VALID_METHODS", _VALID_METHODS | {"newfit"})
    with pytest.raises(LookupError, match=r"method='newfit' is accepted but has no fit builder"):
        _build_fit("newfit", _request())
