from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest

from kaimei import Kaimei
from kaimei._objects import APIObject
from kaimei.images import image_feature, load_image, overlay_heatmap

Image = pytest.importorskip("PIL.Image")


def _photo(width: int = 500, height: int = 375, seed: int = 0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.stack([xs / width, ys / height, rng.random((height, width))], axis=-1)
    return Image.fromarray((pixels * 255).astype(np.uint8))


def test_load_image_matches_torchvision_resize_and_center_crop():
    transforms = pytest.importorskip("torchvision.transforms")
    for size in [(500, 375), (375, 500), (224, 300), (640, 640)]:
        photo = _photo(*size)
        expected = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])(photo)
        loaded = np.asarray(load_image(photo, 224))
        assert loaded.shape == (224, 224, 3)
        np.testing.assert_allclose(loaded, expected.numpy().transpose(1, 2, 0), atol=6e-5)


def test_load_image_reads_size_channels_and_units_from_the_model():
    model = APIObject(
        {
            "slug": "digits",
            "features": [
                {"name": "age", "type": "numeric"},
                {"name": "scan", "type": "image", "height": 8, "width": 6, "channels": 1, "value_range": [0, 255]},
            ],
        }
    )
    assert image_feature(model)["name"] == "scan"
    scan = np.asarray(load_image(_photo(60, 80), model=model))
    assert scan.shape == (8, 6, 1)
    assert scan.max() > 1.5 and scan.max() <= 255
    # Explicit arguments win over the model's feature.
    assert np.asarray(load_image(_photo(), (4, 5), model=model, channels=3)).shape == (4, 5, 3)
    with pytest.raises(ValueError, match="no image feature"):
        image_feature({"features": [{"name": "age", "type": "numeric"}]})


def test_load_image_reads_paths_and_bytes_and_applies_exif_rotation(tmp_path):
    photo = _photo(40, 20)
    png = tmp_path / "photo.png"
    photo.save(png)
    from_path = np.asarray(load_image(png))
    assert from_path.shape == (20, 40, 3)
    np.testing.assert_allclose(from_path, np.asarray(photo) / 255.0, atol=6e-5)
    assert load_image(png.read_bytes()) == load_image(str(png))

    # A phone photo stored landscape with "rotate 90°" in its EXIF is shown (and read) upright.
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    photo.save(buffer, format="JPEG", exif=exif)
    assert np.asarray(load_image(buffer.getvalue())).shape == (40, 20, 3)


def test_load_image_accepts_arrays_in_either_layout():
    rng = np.random.default_rng(1)
    hwc = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
    np.testing.assert_allclose(np.asarray(load_image(hwc)), hwc / 255.0, atol=6e-5)
    chw = rng.random((3, 16, 12))
    np.testing.assert_allclose(np.asarray(load_image(chw, (16, 12))), chw.transpose(1, 2, 0), atol=6e-5)
    gray = np.asarray(load_image(rng.random((16, 12)), channels=3))
    assert gray.shape == (16, 12, 3) and np.allclose(gray[..., 0], gray[..., 2])
    # A different size goes through the same resize + crop as a photo.
    assert np.asarray(load_image(hwc, 8)).shape == (8, 8, 3)
    assert np.asarray(load_image(hwc, (4, 9), crop="stretch")).shape == (4, 9, 3)
    with pytest.raises(ValueError, match="crop"):
        load_image(hwc, crop="pad")


def test_overlay_heatmap_tints_support_cyan_and_opposition_pink():
    gray = np.full((4, 4, 3), 0.5).tolist()
    overlay = np.asarray(overlay_heatmap(gray, [[1.0, -1.0], [0.0, 0.0]]))
    assert overlay.shape == (4, 4, 3)
    assert overlay[0, 0, 2] > overlay[0, 0, 0]  # cyan: blue over red
    assert overlay[0, 3, 0] > overlay[0, 3, 2]  # pink: red over blue
    assert (overlay[3, :] == 128).all()  # no heat, no tint
    on_photo = overlay_heatmap(_photo(40, 30), [[0.5]])
    assert on_photo.size == (40, 30)


def test_numpy_inputs_are_sent_as_compact_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(201, json={"object": "explanation", "id": "exp_1", "options": {"heatmap": [[0.0]]}})

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(handler))
    photo = np.zeros((2, 2, 3), dtype=np.float32)
    client.explanations.create(
        model="cats-dogs-photos", input={"photo": photo, "weight": np.float32(1.5)}, method="gradcam"
    )
    body = json.loads(seen["body"])
    assert body["input"] == {"photo": photo.tolist(), "weight": 1.5}
    assert b", " not in seen["body"] and b": " not in seen["body"]
    with pytest.raises(TypeError, match="not JSON serializable"):
        client.explanations.create(model="m", input={"when": object()})
