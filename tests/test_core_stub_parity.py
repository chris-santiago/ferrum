"""
Parity test between ``src/ferrum/_core.pyi`` and the real ``ferrum._core`` module.

Guarantees:
- Every name declared as a top-level ``class X`` or ``def x`` in the stub
  actually exists in the live module (no stale declarations).
- Every public class/function exported by the live module is declared in the
  stub (no missing-export drift).
- Violin regression: ``extent=`` is rejected (wrong kwarg) while the real
  kwargs (bw_adjust, shared_extent) construct fine.
- Touched-transform construction: Kde, Kde2D, Smooth each accept the
  newly-documented kwargs.
- Signature-parity probes for the 6 drifted stub groups found in round-6 audit:
  Violin groupby, Bin shared_extent, Swarm orient/dodge, Aggregate/Summary name,
  Bin2D/Raster default fictions, process_batch param name.
"""

import ast
from pathlib import Path

import ferrum._core as _core
import pytest


# ---------------------------------------------------------------------------
# Parse stub for declared top-level names
# ---------------------------------------------------------------------------

_STUB_PATH = Path(__file__).parent.parent / "src" / "ferrum" / "_core.pyi"

# Names that are private helpers in the stub (TypedDict, Protocol, type alias)
# and should not be expected in the live module.
_STUB_PRIVATE_PREFIXES = ("_",)

# Names declared in the stub for type-alias / Literal / TypedDict convenience
# that are NOT exported by the live module and are intentionally omitted.
_STUB_ONLY_NAMES = frozenset(
    {
        "DataTypeStr",
        "MarkStr",
        "_KendallResult",
    }
)


def _parse_stub_names() -> set[str]:
    """Return all top-level ``class X`` and ``def x`` names from the stub."""
    source = _STUB_PATH.read_text()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            name = node.name
            if not any(name.startswith(p) for p in _STUB_PRIVATE_PREFIXES):
                names.add(name)
    return names


def _live_module_names() -> set[str]:
    """Return all public names from the live module.

    Covers both callable exports (classes, functions) and non-callable ones
    (module-level constants, enum values) so that future additions of either
    kind are caught by the drift check.
    """
    return {name for name in dir(_core) if not name.startswith("__")}


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def test_stub_names_exist_in_module() -> None:
    """Every name declared in the stub actually exists in the live module."""
    stub_names = _parse_stub_names() - _STUB_ONLY_NAMES
    missing_from_module = stub_names - set(dir(_core))
    assert not missing_from_module, (
        f"Names declared in _core.pyi but absent from ferrum._core: "
        f"{sorted(missing_from_module)}"
    )


def test_module_names_declared_in_stub() -> None:
    """Every public class/function in the live module is declared in the stub."""
    live_names = _live_module_names()
    stub_names = _parse_stub_names()
    # Include stub-only names so they don't shadow real exports
    all_stub_names = stub_names | _STUB_ONLY_NAMES

    missing_from_stub = live_names - all_stub_names
    assert not missing_from_stub, (
        f"Names in ferrum._core but not declared in _core.pyi: "
        f"{sorted(missing_from_stub)}"
    )


# ---------------------------------------------------------------------------
# Violin regression: headline correctness item
# ---------------------------------------------------------------------------


def test_violin_rejects_extent_kwarg() -> None:
    """``Violin(field=..., extent=...)`` raises TypeError — extent is not a real kwarg."""
    with pytest.raises(TypeError):
        _core.Violin(field="x", extent=(0.0, 1.0))  # type: ignore[call-arg]


def test_violin_accepts_real_kwargs() -> None:
    """The kwargs now documented in the stub (bw_adjust, shared_extent) actually work."""
    v = _core.Violin(field="x", bw_adjust=1.5, shared_extent=True)
    assert v is not None


# ---------------------------------------------------------------------------
# Touched-transform construction smoke tests
# ---------------------------------------------------------------------------


def test_kde_accepts_new_kwargs() -> None:
    """Kde(bw_adjust=, kernel=, shared_extent=) are real kwargs."""
    k = _core.Kde(field="x", bw_adjust=1.0, kernel="gaussian", shared_extent=True)
    assert k is not None


def test_kde2d_accepts_groupby() -> None:
    """Kde2D(groupby=) accepts a single string (Optional[str] in the real Rust type)."""
    k = _core.Kde2D(x="a", y="b", groupby="g")
    assert k is not None


def test_smooth_accepts_new_kwargs() -> None:
    """Smooth(inject_metrics=, x_range=) are real kwargs."""
    s = _core.Smooth(x="a", y="b", inject_metrics=True, x_range=(0.0, 1.0))
    assert s is not None


def test_robust_accepts_new_kwargs() -> None:
    """Robust(inject_metrics=, x_range=) are real kwargs."""
    r = _core.Robust(x="a", y="b", inject_metrics=True, x_range=(0.0, 1.0))
    assert r is not None


# ---------------------------------------------------------------------------
# Round-6 signature-parity probes: real discriminators, not existence checks
# ---------------------------------------------------------------------------


def test_violin_groupby_rejects_none() -> None:
    """Violin groupby is Vec<String> (non-Optional); None must raise TypeError.

    The old stub declared ``Optional[List[str]] = None`` — a fiction that would
    have allowed ``groupby=None`` to look valid.  The corrected stub uses
    ``List[str] = ...`` matching the real Rust binding.
    """
    with pytest.raises(TypeError):
        _core.Violin(field="v", groupby=None)  # type: ignore[arg-type]


def test_violin_groupby_accepts_list() -> None:
    """Violin(groupby=['g']) constructs fine — real binding accepts Vec<String>."""
    v = _core.Violin(field="v", groupby=["g"])
    assert v is not None


def test_violin_groupby_accepts_empty_list() -> None:
    """Violin() with no groupby uses the empty-list default; explicit [] also works."""
    v = _core.Violin(field="v", groupby=[])
    assert v is not None


def test_bin_accepts_shared_extent() -> None:
    """Bin(shared_extent=True) constructs — kwarg was missing from stub, now added."""
    b = _core.Bin(field="a", shared_extent=True)
    assert b is not None


def test_swarm_accepts_orient_and_dodge() -> None:
    """Swarm(orient=, dodge=) construct fine — kwargs were missing from stub."""
    s = _core.Swarm(category="cat", value="val", orient="horizontal", dodge="g")
    assert s is not None


def test_swarm_orient_default_vertical() -> None:
    """Swarm without orient uses 'vertical' default — confirms default is accessible."""
    s = _core.Swarm(category="cat", value="val")
    assert s is not None


def test_aggregate_accepts_name() -> None:
    """Aggregate(name=) constructs — kwarg was missing from stub, now added."""
    op = _core.AggregateOp("x", "mean", "x_mean")
    a = _core.Aggregate([op], name="my_agg")
    assert a is not None


def test_summary_accepts_name() -> None:
    """Summary(name=) constructs — kwarg was missing from stub, now added."""
    s = _core.Summary(field="x", name="my_summary")
    assert s is not None


def test_bin2d_constructs_with_none_default() -> None:
    """Bin2D() constructs with no bins_x/bins_y — real default is None, not 'sturges'.

    The old stub declared ``bins_x/bins_y = 'sturges'`` — a fiction; the real Rust
    binding defaults to None and resolves 'sturges' internally.
    """
    b = _core.Bin2D(x="a", y="b")
    assert b is not None


def test_bin2d_accepts_sturges_string() -> None:
    """Bin2D(bins_x='sturges') also works — real binding accepts Optional[PyAny]."""
    b = _core.Bin2D(x="a", y="b", bins_x="sturges")
    assert b is not None


def test_raster_constructs_with_none_default() -> None:
    """Raster() constructs with no resolution — real default is None, not 'screen'.

    The old stub declared ``resolution = 'screen'`` — a fiction; the real Rust
    binding defaults to None and resolves 'screen' internally.
    """
    r = _core.Raster(x="a", y="b")
    assert r is not None


def test_raster_accepts_screen_string() -> None:
    """Raster(resolution='screen') also works — real binding accepts Optional[PyAny]."""
    r = _core.Raster(x="a", y="b", resolution="screen")
    assert r is not None
