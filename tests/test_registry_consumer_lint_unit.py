"""Unit tests for ``tests/_registry_consumer_lint.py`` (wave 6B `#164`).

Moved out of ``tests/test_registry_output_column_consumers.py``: the
synthetic-module RED proof that pins every shape the classifier claims to
recognize (moved verbatim from that file's old ``:1012``), plus one isolated
fixture per :data:`lint.ACCOUNTING` rule so a rule's predicate breaking shows
up here, not only as a drop somewhere inside the sweep's overall counts.

Each per-rule fixture writes a tiny synthetic module under ``tmp_path``,
runs the real lint module against it, and asserts the one literal it plants
lands under the expected label -- discriminating, because deleting that
rule's branch in ``_classify_module`` makes the literal fall through to a
*different* label, failing the assertion. Precedence between two rules that
could both claim the same node (a channel position inside a tuple-unpack or
loop container, which would default to ``vocabulary-member`` without the
container-resolution ``column-reference`` check running first) is a
different property -- each isolated fixture plants its literal in only one
such position, so it is the moved RED proof
(``test_the_classifier_labels_every_shape_it_claims``, see its
``tuple_unpacked``/``chained_assign``/``loop_target`` cases) that covers
ordering/precedence, not these fixtures.
"""

from __future__ import annotations

from tests import _registry_consumer_lint as lint


def _labels_for(tmp_path, source: str, module_name: str = "synthetic.py") -> dict[str, set[str]]:
    """Write *source* under tmp_path, classify it, and index by label."""
    (tmp_path / module_name).write_text(source)
    findings = lint.run(tmp_path)
    by_label: dict[str, set[str]] = {}
    for occurrence in findings.outer_bound:
        by_label.setdefault(occurrence.label, set()).add(
            f"{occurrence.function}:{occurrence.literal}"
        )
    return by_label


def test_the_classifier_labels_every_shape_it_claims(tmp_path):
    """RED proof, in each shape a real consumer is written in.

    Every function below mirrors a site that exists (or existed pre-conversion)
    in ``src/ferrum``; the comment names it. The controls at the end must NOT be
    labelled ``column-reference``, so the classifier is shown to discriminate
    rather than to flag every string it meets.
    """
    source = '''
from dataclasses import replace
from ferrum._layer import _Layer, MarkDesugarResult
from ferrum.encoding import X, Y
from ferrum.encoding import X as _X_ecdf
from ferrum._registry import registry_output_columns, registry_secondary_columns

_SEGMENTS_KEY = "segments"

def inline_dict():
    """marks/composite.py desugar_boxplot, pre-conversion."""
    return _Layer(mark="line", encoding={"y": "ci_lower"}, data_source="smooth")

def name_bound_dict():
    """marks/statistical.py desugar_histogram's encoding_remap."""
    encoding_remap = {"x": "bin_start"}
    return MarkDesugarResult(mark="bar", remap=encoding_remap)

def helper_built_encoding():
    """marks/composite.py desugar_boxplot's enc(...) helper."""
    def enc(col):
        return {"y": col}
    return _Layer(mark="rule", encoding=enc("lower_whisker"), data_source="box")

def aliased_channel_class():
    """plots/distribution.py displot(kind='ecdf'), pre-conversion."""
    enc = {}
    enc["x"] = _X_ecdf("bin_start", title="size")
    enc["y"] = "count"
    return enc

def ternary_encoding(horizontal):
    """marks/composite.py desugar_boxen's ternary encoding dict."""
    return _Layer(
        mark="rule",
        encoding={"x": "lower", "y": "q1"} if horizontal else {"x": "q3", "y": "lower"},
        data_source="lv_depth_1",
    )

def name_bound_to_a_channel_object(cat_sort):
    """marks/composite.py desugar_boxen's group_enc_x."""
    group_enc_x = X("group", sort=cat_sort) if cat_sort is not None else "group"
    return _Layer(mark="bar", encoding={"x": group_enc_x}, data_source="lv_outliers")

def replace_built_layer(layer, ch):
    """_layer_transforms.py _resolve_layer_bins."""
    return replace(layer, encoding={"x": ch.__class__("bin_start")}, data_source=ch.name)

def chart_level_encode(chart):
    """plots/matrix.py jointplot(kind='hist') and plots/regression.py's Q-Q panel."""
    return chart.encode(x="x_lo", y=Y("sample", title="Sample quantiles"))

def converted_argument_and_read():
    cols = registry_output_columns("smooth", ["ci_upper"])
    return _Layer(mark="line", encoding={"y": cols["ci_upper"]}, data_source="smooth")

def converted_secondary_argument_and_read(name):
    """plots/matrix.py's dendrogram layers: a named SECONDARY batch, whose
    key is a module constant because it also builds the data_source."""
    cols = registry_secondary_columns("linkage", _SEGMENTS_KEY, ["x2"])
    return _Layer(
        mark="segment",
        encoding={"x2": cols["x2"]},
        data_source=f"{name}_{_SEGMENTS_KEY}",
    )

def tuple_unpacked():
    """Round-5: a same-arity tuple unpack feeding a channel position."""
    lo, hi = "x_lo", "x_hi"
    return _Layer(mark="rect", encoding={"x": lo, "x2": hi}, data_source="bin_2d")

def chained_assign():
    """Round-5: a chained ``a = b = value`` assignment feeding a channel."""
    a = b = "bin_start"
    return _Layer(mark="bar", encoding={"x": a, "y": b}, data_source="bin")

def loop_target():
    """Round-5: desugar_violin's pre-round-4 shape -- a for-loop variable
    bound to a literal container, used in a channel position on each pass."""
    out = {}
    for name in ("q1", "median", "q3"):
        out["x"] = name
    return _Layer(mark="rule", encoding=out, data_source="box_stats")

def controls(other, axis):
    """None of these names a transform output column."""
    picked = other["ci_lower"]
    for name in ("q1", "median"):
        picked = picked or name
    if axis == "density":
        picked = other.get("theoretical")
    return _Layer(mark="text", mark_kwargs={"align": "right"}, encoding={"text": picked})
'''
    by_label = _labels_for(tmp_path, source, "synthetic_consumers.py")

    assert by_label["column-reference"] == {
        "inline_dict:ci_lower",
        "name_bound_dict:bin_start",
        "helper_built_encoding:lower_whisker",
        "aliased_channel_class:bin_start",
        "aliased_channel_class:count",
        "ternary_encoding:lower",
        "ternary_encoding:q1",
        "ternary_encoding:q3",
        "name_bound_to_a_channel_object:group",
        "replace_built_layer:bin_start",
        "chart_level_encode:x_lo",
        "chart_level_encode:sample",
        "tuple_unpacked:x_lo",
        "tuple_unpacked:x_hi",
        "chained_assign:bin_start",
        "loop_target:q1",
        "loop_target:median",
        "loop_target:q3",
    }
    assert by_label["converted-argument"] == {
        "converted_argument_and_read:ci_upper",
        "converted_secondary_argument_and_read:x2",
    }
    assert by_label["converted-lookup-read"] == {
        "converted_argument_and_read:ci_upper",
        "converted_secondary_argument_and_read:x2",
    }
    # The controls land in the non-column bucket that describes them. (These
    # are subset assertions because the module's own channel keys -- the "x"
    # and "y" of every encoding dict above -- are themselves declared output
    # names and so are mapping-key occurrences too.)
    assert "controls:ci_lower" in by_label["mapping-key"]
    assert {"controls:q1", "controls:median"} <= by_label["vocabulary-member"]
    assert "controls:density" in by_label["predicate-operand"]
    assert "controls:theoretical" in by_label["call-argument"]
    assert "controls:right" in by_label["non-channel-dict-value"]
    found = lint.run(tmp_path).outer_bound
    assert not [o for o in found if o.label not in lint.LABELS]


# ---------------------------------------------------------------------------
# One isolated fixture per lint.ACCOUNTING rule. Each plants exactly one
# declared-name literal in the one syntactic position the rule describes, so
# breaking that rule's predicate flips the literal to a different label and
# fails the assertion below. These do not exercise precedence between two
# rules that could both match the same node (see the module docstring); that
# is the RED proof's job.
# ---------------------------------------------------------------------------


def test_rule_converted_argument_and_lookup_read(tmp_path):
    source = """
from ferrum._registry import registry_output_columns

def read():
    cols = registry_output_columns("smooth", ["ci_upper"])
    return cols["ci_upper"]
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["converted-argument"] == {"read:ci_upper"}
    assert by_label["converted-lookup-read"] == {"read:ci_upper"}


def test_rule_column_reference(tmp_path):
    source = """
def encode():
    return {"x": "bin_start"}
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["column-reference"] == {"encode:bin_start"}


def test_rule_mapping_key(tmp_path):
    source = """
def read(other):
    return other["ci_lower"]
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["mapping-key"] == {"read:ci_lower"}


def test_rule_vocabulary_member(tmp_path):
    source = """
def sweep():
    out = []
    for name in ("q1", "median"):
        out.append(name)
    return out
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["vocabulary-member"] == {"sweep:q1", "sweep:median"}


def test_rule_predicate_operand(tmp_path):
    source = """
def check(axis):
    if axis == "density":
        return True
    return False
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["predicate-operand"] == {"check:density"}


def test_rule_non_channel_keyword(tmp_path):
    source = """
def label_thing(thing):
    return thing.set(title="sample")
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["non-channel-keyword"] == {"label_thing:sample"}


def test_rule_non_channel_dict_value(tmp_path):
    source = """
def props():
    return {"align": "right"}
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["non-channel-dict-value"] == {"props:right"}


def test_rule_call_argument(tmp_path):
    source = """
def read(other):
    return other.get("theoretical")
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["call-argument"] == {"read:theoretical"}


def test_rule_local_binding(tmp_path):
    source = """
_DEFAULT_COUNT = "count"
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["local-binding"] == {"<module>:count"}


def test_rule_message_text(tmp_path):
    source = """
def build(prefix):
    return f"{prefix}count"
"""
    by_label = _labels_for(tmp_path, source)
    assert by_label["message-text"] == {"build:count"}
