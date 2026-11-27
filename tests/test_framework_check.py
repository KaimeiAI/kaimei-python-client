"""A framework given for a model object that says beyond doubt which one it is made in must be that one."""

import pytest

from kaimei.structure import StructureError, _check_framework, extract_structure


def _made_in(module: str):
    return type("Net", (), {"__module__": module})()


def test_a_framework_the_model_is_not_made_in_is_refused_before_anything_is_read():
    with pytest.raises(StructureError) as raised:
        extract_structure(_made_in("torch.nn.custom"), framework="tensorflow")
    assert raised.value.code == "framework_mismatch"
    assert raised.value.param == "framework"
    assert "pytorch" in str(raised.value) and "Leave framework out" in str(raised.value)
    with pytest.raises(StructureError):
        _check_framework(_made_in("xgboost.sklearn"), "sklearn")


@pytest.mark.parametrize(
    "module, framework",
    [
        ("torch.nn.custom", None),
        ("torch.nn.custom", "pytorch"),
        ("torch.nn.custom", "Torch"),
        ("torch.nn.custom", "custom"),
        ("keras.src.models", "tensorflow"),
        ("tensorflow.python.module", "keras"),
        # A scikit-learn object can wrap any other framework's model: what it is given is taken.
        ("sklearn.pipeline", "xgboost"),
        # Nothing to go by.
        ("my_package.model", "jax"),
    ],
)
def test_a_framework_that_fits_or_cannot_be_checked_is_taken(module, framework):
    _check_framework(_made_in(module), framework)
