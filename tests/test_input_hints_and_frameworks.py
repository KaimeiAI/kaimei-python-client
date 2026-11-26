"""Two small readings that guessed: a picture's layout given to an input that is no picture, and a "torch" by name.

A ``layout`` hint on an input whose rows are not height, width and channels failed deep in the importer with
"list index out of range" and named no argument. And any class from a package whose name began with "torch" was
taken for a PyTorch module, as a package beginning with "jax" was taken for JAX.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaimei.onnx_graph import OnnxImportError, _composite_feature
from kaimei.structure import _is_torch_module


@pytest.mark.parametrize("sample", [[8], [4, 8], [1, 2, 3, 4], []])
def test_a_pictures_layout_on_an_input_that_is_no_picture_names_the_argument(sample: list) -> None:
    with pytest.raises(OnnxImportError) as caught:
        _composite_feature("f", sample, {"feature": "f", "layout": "nchw"}, np.dtype("float32"), {})
    assert caught.value.param == "inputs"
    assert "height, width and channels" in str(caught.value)


def test_a_pictures_layout_on_a_picture_is_read_as_before() -> None:
    first = _composite_feature("photo", [3, 4, 5], {"feature": "photo", "layout": "nchw"}, np.dtype("float32"), {})
    last = _composite_feature("photo", [4, 5, 3], {"feature": "photo", "layout": "nhwc"}, np.dtype("float32"), {})
    for feature in (first, last):
        assert (feature["type"], feature["height"], feature["width"], feature["channels"]) == ("image", 4, 5, 3)


def _instance_from(package: str, **attributes: object) -> object:
    kind = type("Model", (), dict(attributes))
    kind.__module__ = package
    return kind()


def test_only_pytorchs_own_packages_make_a_model_a_pytorch_module_by_name() -> None:
    assert not _is_torch_module(_instance_from("torchy_models.nets"))
    assert not _is_torch_module(_instance_from("torchlike"))
    assert _is_torch_module(_instance_from("torch.nn.modules.linear"))
    assert _is_torch_module(_instance_from("torchvision.models.resnet"))
    # A module of any package is still one by what it has, as before.
    assert _is_torch_module(_instance_from("my_models", named_parameters=lambda self: [], forward=lambda self: None))
