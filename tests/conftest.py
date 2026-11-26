"""Shared test setup.

The test files that came before the ONNX route describe Kaimei's own readers: what they read exactly, what
they draw, and what they refuse. With ``via="auto"`` a model those readers can only draw is now converted
to ONNX when the converter for its framework happens to be installed, which is what the newer test files
cover. So that the older files say the same thing whether or not ``onnx``, ``torch`` or ``skl2onnx`` is
installed beside them, they run with no conversion available: exactly the case of an environment that has
only this package's own extras.
"""

from __future__ import annotations

import pytest

#: Test modules written for the native readers alone.
NATIVE_READERS_ONLY = {
    "test_activations",
    "test_actuals",
    "test_client",
    "test_display",
    "test_drawn_models",
    "test_honest_uploads",
    "test_image_helpers",
    "test_images",
    "test_live",
    "test_organizations",
    "test_projects",
    "test_scale",
    "test_tabular",
    "test_upload",
}


@pytest.fixture(autouse=True)
def _native_readers_only(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1] in NATIVE_READERS_ONLY:
        from kaimei import graphs

        monkeypatch.setattr(graphs, "via_onnx", lambda *args, **kwargs: None)
    yield
