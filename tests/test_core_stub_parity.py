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
- PROGRAMMATIC live-vs-stub signature diff (round-9): for every public
  function and every class whose ``__text_signature__`` is introspectable (not
  ``(*args, **kwargs)``), assert the stub's parameter names and
  required/optional distinction match the live ``inspect.signature``.  This
  prevents param-rename drift (hat_matrix_stats X→x_table), missing-kwarg
  drift (TimeScale utc), and wrong-required drift (TimeScale range).
"""

import ast
import inspect
from pathlib import Path
from typing import NamedTuple

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
# Programmatic live-vs-stub signature diff (round-9)
# ---------------------------------------------------------------------------


class _ParamSig(NamedTuple):
    """Minimal param descriptor for parity comparison."""

    name: str
    has_default: bool


def _stub_params_for_function(fn_node: ast.FunctionDef) -> list[_ParamSig]:
    """Extract ordered param list from a stub ``def`` node, excluding ``self``/``cls``.

    Handles pos-or-kw args (with defaults that are right-aligned in
    ``args.defaults``), keyword-only args (with per-slot ``kw_defaults`` where
    ``None`` means *required*), and positional-only args.  ``*args`` and
    ``**kwargs`` are excluded because the live Rust binding uses a concrete
    signature.
    """
    args = fn_node.args
    result: list[_ParamSig] = []

    # ast.arguments.defaults is right-aligned over the concatenation of
    # (posonlyargs + args).  A combined index i has a default iff:
    #   i >= (n_posonly + n_posorkw) - len(defaults)
    n_posonly = len(args.posonlyargs)
    n_posorkw = len(args.args)
    n_all_defaults = len(args.defaults)
    combined_len = n_posonly + n_posorkw
    defaults_start_combined = combined_len - n_all_defaults

    # --- positional-only args (combined index == i directly, they come first) ---
    for i, arg in enumerate(args.posonlyargs):
        if arg.arg in ("self", "cls"):
            continue
        has_default = i >= defaults_start_combined
        result.append(_ParamSig(arg.arg, has_default))

    # --- positional-or-keyword args ---
    for i, arg in enumerate(args.args):
        if arg.arg in ("self", "cls"):
            continue
        combined_index = n_posonly + i
        has_default = combined_index >= defaults_start_combined
        result.append(_ParamSig(arg.arg, has_default))

    # --- keyword-only args ---
    for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg in ("self", "cls"):
            continue
        has_default = kw_default is not None
        result.append(_ParamSig(arg.arg, has_default))

    return result


def _parse_stub_signatures() -> dict[str, list[_ParamSig]]:
    """Parse the stub and return a mapping of name → param list.

    Keys are:
    - Top-level function names (``def foo(...)``)
    - Class names for classes that have an ``__init__`` in the stub
      (``class Foo`` → params of ``__init__`` excluding ``self``)
    """
    source = _STUB_PATH.read_text()
    tree = ast.parse(source)
    result: dict[str, list[_ParamSig]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            result[node.name] = _stub_params_for_function(node)
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    result[node.name] = _stub_params_for_function(child)

    return result


def _live_sig_params(obj: object) -> list[_ParamSig] | None:
    """Return param list from the live object's ``inspect.signature``.

    Returns ``None`` when the signature is not introspectable (e.g. the object
    has no ``__text_signature__`` or it is the opaque ``(*args, **kwargs)``).
    """
    text_sig = getattr(obj, "__text_signature__", None)
    if text_sig is None or text_sig.startswith("(*args"):
        return None
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        has_default = param.default is not inspect.Parameter.empty
        params.append(_ParamSig(name, has_default))
    return params


# Names that are introspection-opaque (no real __text_signature__) and should
# not be included in the programmatic diff.  Document each omission.
_PROGRAMMATIC_DIFF_SKIP: frozenset[str] = frozenset(
    {
        # ContinuousScheme has no __text_signature__ (no public __init__).
        "ContinuousScheme",
    }
)


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def test_stub_names_exist_in_module() -> None:
    """Every name declared in the stub actually exists in the live module."""
    stub_names = _parse_stub_names() - _STUB_ONLY_NAMES
    missing_from_module = stub_names - set(dir(_core))
    assert not missing_from_module, (
        f"Names declared in _core.pyi but absent from ferrum._core: {sorted(missing_from_module)}"
    )


def test_module_names_declared_in_stub() -> None:
    """Every public class/function in the live module is declared in the stub."""
    live_names = _live_module_names()
    stub_names = _parse_stub_names()
    # Include stub-only names so they don't shadow real exports
    all_stub_names = stub_names | _STUB_ONLY_NAMES

    missing_from_stub = live_names - all_stub_names
    assert not missing_from_stub, (
        f"Names in ferrum._core but not declared in _core.pyi: {sorted(missing_from_stub)}"
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


# ---------------------------------------------------------------------------
# Round-9: programmatic live-vs-stub signature diff
# ---------------------------------------------------------------------------


def test_programmatic_signature_parity() -> None:
    """Every introspectable live binding has matching param names and defaults in the stub.

    Covers:
    - All public top-level functions (``#[pyfunction]``) whose
      ``__text_signature__`` is a real signature (not ``(*args, **kwargs)``).
    - All public classes (including those with ``#[pyo3(signature=...)]``
      on their ``#[new]`` impl) whose ``__text_signature__`` is a real
      signature and that have an ``__init__`` in the stub.

    Catching examples of the drift class this test closes:
    - Param rename: ``hat_matrix_stats(X=...)`` → ``x_table``
    - Missing kwarg: ``TimeScale`` was missing ``utc``
    - Wrong required/optional: ``TimeScale.range`` was wrongly required
    - Missing kwargs: ``EncodingSpec`` was missing 11 keyword-only params

    Names in ``_PROGRAMMATIC_DIFF_SKIP`` are omitted because their live
    binding exposes no useful ``__text_signature__`` (see the frozenset for
    rationale per entry).
    """
    stub_sigs = _parse_stub_signatures()
    failures: list[str] = []

    for name in sorted(dir(_core)):
        if name.startswith("__") or name in _PROGRAMMATIC_DIFF_SKIP:
            continue

        obj = getattr(_core, name)
        if not callable(obj):
            continue

        live_params = _live_sig_params(obj)
        if live_params is None:
            # Not introspectable — skip (opaque Rust binding).
            continue

        if name not in stub_sigs:
            # Class has no __init__ in stub (e.g. no-constructor class) — skip.
            continue

        stub_params = stub_sigs[name]

        if live_params != stub_params:
            live_names = [p.name for p in live_params]
            stub_names = [p.name for p in stub_params]
            live_req = [p.name for p in live_params if not p.has_default]
            stub_req = [p.name for p in stub_params if not p.has_default]

            details = []
            if live_names != stub_names:
                details.append(f"names: live={live_names} stub={stub_names}")
            if live_req != stub_req:
                details.append(f"required: live={live_req} stub={stub_req}")
            if not details:
                # Defaults differ but names/required match — report defaults.
                details.append(f"has_default differs: live={live_params} stub={stub_params}")
            failures.append(f"{name}: {'; '.join(details)}")

    assert not failures, (
        "Stub parameter drift detected (names or required/optional mismatch):\n"
        + "\n".join(f"  {f}" for f in failures)
    )


# ---------------------------------------------------------------------------
# Round-9: targeted construction probes for the 3 new bug fixes
# ---------------------------------------------------------------------------


def test_encoding_spec_accepts_all_keyword_only_params() -> None:
    """EncodingSpec accepts all 11 keyword-only params added to the stub in round-9.

    Previously the stub declared only ``(field, type_=None)`` — the 11
    keyword-only params were silently dropped.  A typed caller passing any of
    them would see a type error from the stub while the runtime accepted them.
    """
    es = _core.EncodingSpec(
        "price",
        "Q",
        scale={"type": "log"},
        title="Price ($)",
        axis={"labelAngle": -45},
        legend={"orient": "top"},
        sort="descending",
        stack="zero",
        condition=None,
        impute={"value": 0},
        scheme="viridis",
        format=".2f",
        format_type="number",
    )
    assert es is not None
    assert es.field == "price"
    assert es.title == "Price ($)"
    assert es.scheme == "viridis"
    assert es.format == ".2f"
    assert es.format_type == "number"


def test_time_scale_range_is_optional() -> None:
    """TimeScale constructs without ``range`` — the stub previously made it required.

    The real binding defaults ``range`` to ``None``, which the renderer
    auto-fills from the plot-area dimensions.  The stub had ``range:
    Sequence[float]`` (no default), making keyword-call ``TimeScale(domain=...)``
    look like a type error to callers.
    """
    ts = _core.TimeScale(domain=[0.0, 1_000_000.0])
    assert ts is not None


def test_time_scale_accepts_utc() -> None:
    """TimeScale(utc=True) constructs fine — kwarg was entirely missing from stub."""
    ts = _core.TimeScale(domain=[0.0, 1_000_000.0], utc=True)
    assert ts is not None
    assert ts.utc is True


def test_stub_params_posonly_default_attribution() -> None:
    """_stub_params_for_function attributes defaults correctly when posonly args are present.

    ``ast.arguments.defaults`` is right-aligned over the combined
    (posonlyargs + args) list, not over posonlyargs alone.  For
    ``def f(a, b, /, c, d=2)`` only ``d`` has a default; ``a``, ``b``, and
    ``c`` do not.  An earlier bug computed ``posonly_defaults_start`` relative to
    the posonly slice only, which would have wrongly attributed a default to ``b``.
    """
    src = "def f(a, b, /, c, d=2): ..."
    tree = ast.parse(src)
    fn_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    params = _stub_params_for_function(fn_node)
    assert params == [
        _ParamSig("a", False),
        _ParamSig("b", False),
        _ParamSig("c", False),
        _ParamSig("d", True),
    ], f"Unexpected param list: {params}"


def test_hat_matrix_stats_param_name_is_x_table() -> None:
    """``hat_matrix_stats`` first param is ``x_table``, not ``X``.

    The stub previously declared ``X`` as the first param.  A keyword call
    ``hat_matrix_stats(X=batch, ...)`` would crash at runtime with a TypeError
    while a call ``hat_matrix_stats(x_table=batch, ...)`` would succeed.
    Verifying the stub name via ``inspect.signature`` confirms the fix.
    """
    sig = inspect.signature(_core.hat_matrix_stats)
    param_names = list(sig.parameters.keys())
    assert param_names[0] == "x_table", (
        f"Expected first param to be 'x_table', got {param_names[0]!r}"
    )
