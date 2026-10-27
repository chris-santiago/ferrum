"""PLOT-09 — the deprecated dispatcher shims are off ``__all__`` but importable.

``shap_chart(kind=...)`` and ``rank_chart(rank=...)`` are deprecated dispatchers
that warn and forward to their split siblings. This pins the advertised-surface
reduction (dropped from ``ferrum.__all__`` and ``ferrum.plots.__all__``) together
with the behavior preservation (still importable, still warn, still forward), so a
future "re-add to ``__all__``" or "drop the warn" cannot pass silently.

The forwarding is asserted by monkeypatching the sibling functions on the shim's
own module, so this test does not depend on the (separately tested) sklearn/SHAP
rendering machinery — only on the warn + dispatch contract.
"""

from __future__ import annotations

import pytest

import ferrum
import ferrum.plots as ferrum_plots


# ---- advertised-surface reduction -------------------------------------------


@pytest.mark.parametrize("name", ["shap_chart", "rank_chart"])
def test_deprecated_dispatcher_dropped_from_all(name: str) -> None:
    """The shims are no longer advertised in either ``__all__``."""
    assert name not in ferrum.__all__
    assert name not in ferrum_plots.__all__


@pytest.mark.parametrize("name", ["shap_chart", "rank_chart"])
def test_deprecated_dispatcher_still_importable(name: str) -> None:
    """Dropping from ``__all__`` does not remove the names — direct import works,
    and both spellings resolve to the same object.
    """
    top = getattr(ferrum, name)
    via_plots = getattr(ferrum_plots, name)
    assert top is via_plots
    assert callable(top)


# ---- behavior preservation: warn + forward ----------------------------------


def test_shap_chart_warns_and_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """``shap_chart(kind="bar")`` warns then delegates to
    ``_shap_bar_chart_from_source`` (the canonical split sibling).
    """
    import ferrum.plots.explanation as expl

    calls: list[str] = []

    def _fake_bar(source, **kwargs):
        calls.append("bar")
        return ferrum.Chart(None).mark_point()

    # Avoid the real SHAP computation: stub the source resolver and the sibling.
    monkeypatch.setattr(expl, "_resolve_source", lambda *a, **k: object())
    monkeypatch.setattr(expl, "_shap_bar_chart_from_source", _fake_bar)

    with pytest.warns(DeprecationWarning, match="shap_chart"):
        result = ferrum.shap_chart(object(), kind="bar")

    assert calls == ["bar"]
    # The forwarded chart speaks as the dispatcher the caller invoked.
    assert result._entry == "shap_chart"


def test_rank_chart_warns_and_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """``rank_chart(rank="1d")`` warns then delegates to ``rank1d_chart``'s body.

    ``rank_chart`` calls the private ``_rank1d_chart_impl`` directly (not the
    public, decorated ``rank1d_chart``), so the result carries one replay
    record voiced as ``rank_chart`` rather than two nested ones.
    """
    import ferrum.plots.ranking as ranking

    calls: list[str] = []

    def _fake_rank1d_impl(entry, *args, **kwargs):
        calls.append(entry)
        return ferrum.Chart(None).mark_point()

    monkeypatch.setattr(ranking, "_rank1d_chart_impl", _fake_rank1d_impl)

    with pytest.warns(DeprecationWarning, match="rank_chart"):
        result = ferrum.rank_chart(object(), rank="1d")

    assert calls == ["rank_chart"]
    assert result._entry == "rank_chart"


@pytest.mark.parametrize(
    "func_name, bad_kwargs, message_fragment",
    [
        ("shap_chart", {"kind": "nonsense"}, "kind"),
        ("rank_chart", {"rank": "nonsense"}, "rank"),
    ],
)
def test_deprecated_dispatcher_rejects_unknown_kind(
    func_name: str, bad_kwargs: dict, message_fragment: str
) -> None:
    """An unknown dispatch key still raises ``ValueError`` (tail-error preserved)."""
    func = getattr(ferrum, func_name)
    with pytest.raises(ValueError, match=message_fragment):
        func(object(), **bad_kwargs)
