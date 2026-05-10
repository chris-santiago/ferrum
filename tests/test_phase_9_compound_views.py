"""Phase 9 compound view + Repeat sentinel tests."""
import pytest

import ferrum as fe
from ferrum import Repeat


class TestRepeatSentinel:
    def test_column_row_layer_are_distinct_values(self):
        assert Repeat.column is not Repeat.row
        assert Repeat.row is not Repeat.layer
        assert Repeat.column.field == "column"
        assert Repeat.row.field == "row"
        assert Repeat.layer.field == "layer"

    def test_repr_is_descriptive(self):
        assert repr(Repeat.column) == "Repeat.column"
        assert repr(Repeat.row) == "Repeat.row"

    def test_singleton_identity_across_imports(self):
        # Re-importing should give the same object.
        from ferrum.repeat import Repeat as Repeat2
        assert Repeat.column is Repeat2.column

    def test_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            Repeat.column.field = "row"  # type: ignore

    def test_used_in_encode_serializes_as_dollar_repeat(self):
        # Used via Chart.encode(x=Repeat.column) — RepeatChart expansion converts
        # to a real field, but the bare placeholder must round-trip via to_dict.
        sentinel = Repeat.column
        assert sentinel.to_repeat_dict() == {"$repeat": "column"}
