"""What a reading says when it cannot use something: a value a summary kept at home, and why a conversion is drawn.

An agent given a structure uploaded with ``weights="summary"`` met a fixed input value sent as its shape only and
failed with "Constant ... is not a tensor Kaimei reads". And a reading that fell back from a conversion said only
that the conversion "does not reproduce the model's outputs", even when it had never been compared.
"""

from __future__ import annotations

import pytest

from kaimei.agent import _fixed_value
from kaimei.structure import _not_converted_as_asked


def test_a_fixed_value_a_summary_kept_at_home_says_how_to_give_it() -> None:
    with pytest.raises(ValueError) as caught:
        _fixed_value({"$shape": [1, 4], "dtype": "float32", "stats": {"mean": 0.5}}, "scale")
    said = str(caught.value)
    assert "'scale'" in said and 'weights="full"' in said and "inputs={'scale': lambda row:" in said


def test_a_fixed_value_sent_whole_is_read_as_before() -> None:
    np = pytest.importorskip("numpy")
    value = _fixed_value({"dtype": "float32", "shape": [1, 2], "data": [1.5, 2.5]}, "scale")
    np.testing.assert_allclose(value, [[1.5, 2.5]])
    np.testing.assert_allclose(_fixed_value([3.0], "offset"), [3.0])


def test_a_conversion_that_was_not_compared_is_said_to_be_so() -> None:
    from kaimei import graphs

    converted = {"approximate": True, "summary": f"Read from ONNX. {graphs.NOT_COMPARED} because torch is not installed"}
    error = _not_converted_as_asked(["inputs"], [], converted)
    assert "not reproduce" not in error.message
    assert graphs.NOT_COMPARED[:1].lower() + graphs.NOT_COMPARED[1:] in error.message
    # With nothing more said, the reason stays the general one.
    plain = _not_converted_as_asked(["inputs"], [], {"approximate": True, "summary": "Read from ONNX."})
    assert "the conversion does not reproduce the model's outputs" in plain.message
