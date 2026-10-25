from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from kaimei import Kaimei
from kaimei._objects import APIObject
from kaimei.images import (
    BAR_NEGATIVE,
    BAR_POSITIVE,
    evidence_images,
    explanation_summary,
    focus_mask,
    focus_threshold,
    gaussian_blur,
    heat_alpha,
    heat_color,
    image_feature,
    load_image,
    numbered_regions,
    overlay_heatmap,
    region_mask,
    render_explanation,
    upsample_heatmap,
)

Image = pytest.importorskip("PIL.Image")

FIXTURES = Path(__file__).parent / "fixtures"


def _photo(width: int = 500, height: int = 375, seed: int = 0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.stack([xs / width, ys / height, rng.random((height, width))], axis=-1)
    return Image.fromarray((pixels * 255).astype(np.uint8))


def _bytes(image) -> np.ndarray:
    return np.asarray(image).astype(int)


def _record(method: str = "occlusion", heat=None, output=None, **options) -> APIObject:
    """A synthetic image explanation with the keys the renderer reads."""
    return APIObject(
        {
            "object": "explanation",
            "method": method,
            "output": output
            or {"label": "cat", "probabilities": {"cat": 0.98, "dog": 0.02}, "target_label": "cat", "value": 0.98},
            "attributions": {"upper left": 0.8, "upper right": -0.3, "lower left": 0.1, "lower right": 0.05},
            "options": {"heatmap": heat if heat is not None else [[1.0, -0.4], [0.1, 0.05]], "target": "cat", **options},
        }
    )


# --- loading -------------------------------------------------------------------------------------


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


def test_example_photos_open_at_the_model_size():
    photos = sorted((Path(__file__).parents[1] / "examples" / "photos").glob("*.jpg"))
    assert len(photos) == 7, [p.name for p in photos]
    for photo in photos:
        pixels = np.asarray(load_image(photo, 224))
        assert pixels.shape == (224, 224, 3) and 0.0 <= pixels.min() and pixels.max() <= 1.0
        with Image.open(photo) as opened:
            assert max(opened.size) == 448, photo.name



def test_importing_the_module_loads_neither_matplotlib_nor_pillow():
    code = (
        "import sys, kaimei.images; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'matplotlib', 'PIL'}))"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == "[]"


# --- rendering primitives --------------------------------------------------------------------------


def test_colour_stops_and_alpha_ramp():
    a = np.asarray([0.0, 0.12, 0.42, 0.72, 1.0])
    # Nothing up to 0.12, a smoothstep to 0.66 at 0.72 (r = (a − 0.12) / 0.6), flat above.
    np.testing.assert_allclose(heat_alpha(a), [0.0, 0.0, 0.33, 0.66, 0.66], atol=1e-12)
    np.testing.assert_allclose(heat_alpha(-a), heat_alpha(a))
    np.testing.assert_allclose(heat_alpha(0.27), 0.66 * 0.25**2 * (3 - 0.5), atol=1e-12)
    # Support (v ≥ 0): amber → orange → red, linear between the stops at 0, 0.5 and 1.
    support = heat_color(a)
    np.testing.assert_allclose(
        support,
        [(1.0, 0.86, 0.30), (1.0, 0.7856, 0.2568), (1.0, 0.5996, 0.1488), (0.9692, 0.3652, 0.164), (0.93, 0.13, 0.22)],
        atol=1e-12,
    )
    # Against (v < 0): light blue → blue → indigo.
    against = heat_color(-np.asarray([1e-12, 0.12, 0.42, 0.72, 1.0]))
    np.testing.assert_allclose(
        against,
        [(0.55, 0.65, 1.0), (0.502, 0.602, 1.0), (0.382, 0.482, 1.0), (0.328, 0.362, 0.978), (0.30, 0.25, 0.95)],
        atol=1e-9,
    )
    # Exactly zero counts as support (and is transparent anyway); vectorised over any shape.
    np.testing.assert_allclose(heat_color(0.0), (1.0, 0.86, 0.30))
    assert heat_color(np.zeros((3, 4))).shape == (3, 4, 3)


def test_upsampling_bilinear_uses_half_pixel_centres_and_nearest_stays_blocky():
    heat = [[0.0, 1.0], [0.0, 1.0]]
    bilinear = upsample_heatmap(heat, 4, 4, "bilinear")
    np.testing.assert_allclose(bilinear, np.tile([0.0, 0.25, 0.75, 1.0], (4, 1)))
    nearest = upsample_heatmap(heat, 4, 4, "nearest")
    np.testing.assert_array_equal(nearest, np.tile([0.0, 0.0, 1.0, 1.0], (4, 1)))
    # Rows interpolate the same way, values are clamped to [-1, 1] and the same size is identity.
    np.testing.assert_allclose(upsample_heatmap([[0.0], [2.0]], 4, 1, "bilinear")[:, 0], [0.0, 0.25, 0.75, 1.0])
    np.testing.assert_allclose(upsample_heatmap([[0.3, -0.2], [0.9, 0.1]], 2, 2), [[0.3, -0.2], [0.9, 0.1]])
    with pytest.raises(ValueError, match="kind"):
        upsample_heatmap(heat, 4, 4, "cubic")
    with pytest.raises(ValueError, match="2-D"):
        upsample_heatmap([0.0, 1.0], 4, 4)


def test_focus_outline_and_spotlight_dimming():
    # A 3 × 3 focus square inside a 6 × 6 photo, drawn with nearest upsampling on a flat gray.
    heat = np.zeros((6, 6))
    heat[1:4, 1:4] = 1.0
    gray = np.full((6, 6, 3), 0.5)
    spot = _bytes(overlay_heatmap(gray, heat, method="occlusion", mode="spotlight", threshold=0.5))[..., 0]
    assert spot[2, 2] == 128  # inside, not an edge: the photo itself
    assert spot[0, 0] == 36  # outside: dimmed to 0.28 → floor(0.14 · 255 + 0.5)
    assert (spot == 236).sum() == 8  # inside edge pixels: 0.5 · 0.15 + 0.85
    assert (spot == 222).sum() == 12  # outside pixels touching the square: 0.14 · 0.15 + 0.85
    assert (spot == 128).sum() == 1 and (spot == 36).sum() == 15
    # The spotlight fades in over 0.12 of heat below the threshold; the outline can be turned off.
    half = _bytes(overlay_heatmap(gray, [[0.44]], method="occlusion", mode="spotlight", threshold=0.5, outline=False))
    assert half[0, 0, 0] == int(np.floor(0.5 * (0.28 + 0.72 * 0.5) * 255 + 0.5))
    plain = _bytes(overlay_heatmap(gray, heat, method="occlusion", mode="spotlight", threshold=0.5, outline=False))
    assert set(np.unique(plain)) == {36, 128}
    original = _bytes(overlay_heatmap(gray, heat, method="occlusion", mode="original"))
    assert (original == 128).all()


def test_heatmap_mode_blends_the_ramp_and_rounds_bytes_half_up():
    black = np.zeros((2, 2, 3))
    # Full heat on black: the red top stop at alpha 0.66, floor(x · 255 + 0.5).
    out = _bytes(overlay_heatmap(black, [[1.0]], method="occlusion", outline=False))
    assert out[0, 0].tolist() == [157, 22, 37]  # 0.93, 0.13, 0.22 × 0.66 × 255 = 156.5, 21.9, 37.0
    # |v| = 0.42 on gray: alpha 0.33 over 0.5 · 0.67, in both palettes (hand-computed).
    gray = np.full((1, 2, 3), 0.5)
    both = _bytes(overlay_heatmap(gray, [[0.42, -0.42]], method="lime", outline=False))
    assert both[0, 0].tolist() == [170, 136, 98] and both[0, 1].tolist() == [118, 126, 170]
    # Heat up to 0.12 leaves the photo untouched; 0.5 maps to byte 128, not 127.
    gray = np.full((2, 2, 3), 0.5)
    assert (_bytes(overlay_heatmap(gray, [[0.12]], method="occlusion", outline=False)) == 128).all()
    assert (_bytes(overlay_heatmap(np.full((1, 1, 3), 0.1), [[0.0]], outline=False)) == 26).all()
    # A PIL photo and a value_range photo render the same.
    photo = _photo(8, 6)
    as_pil = _bytes(overlay_heatmap(photo, [[0.7, -0.7]]))
    as_list = _bytes(overlay_heatmap(np.asarray(photo).astype(float), [[0.7, -0.7]], value_range=(0, 255)))
    np.testing.assert_array_equal(as_pil, as_list)


def test_overlay_heatmap_keeps_the_old_call_and_tints_support_warm_and_opposition_blue():
    gray = np.full((4, 4, 3), 0.5).tolist()
    overlay = np.asarray(overlay_heatmap(gray, [[1.0, -1.0], [0.0, 0.0]])).astype(int)
    assert overlay.shape == (4, 4, 3)
    assert overlay[0, 0, 0] > overlay[0, 0, 2] + 60  # support: red over blue
    assert overlay[0, 3, 2] > overlay[0, 3, 0] + 60  # against: blue over red
    assert (overlay[3, :] == 128).all()  # no heat, no tint
    # The glow grows with the heat (deeper and more opaque at the peak).
    glow = np.asarray(overlay_heatmap(np.zeros((4, 4, 3)), [[1.0, 0.4], [0.0, 0.0]], method="gradcam", outline=False))
    glow = glow.astype(int)
    assert glow[0, 0].sum() > glow[0, 3].sum() > glow[3, 3].sum() == 0
    on_photo = overlay_heatmap(_photo(40, 30), [[0.5]], opacity=0.5)  # 0.4 keyword still accepted
    assert on_photo.size == (40, 30)
    with pytest.raises(ValueError, match="mode"):
        overlay_heatmap(gray, [[1.0]], mode="glow")


def test_overlay_heatmap_infers_the_upsampling_from_the_values():
    photo = _photo(12, 12)
    signed = [[1.0, -0.4], [0.1, 0.05]]
    positive = [[1.0, 0.2], [0.0, 0.6]]
    # Any negative value: a region map, drawn blocky like occlusion / LIME (and their threshold).
    np.testing.assert_array_equal(
        _bytes(overlay_heatmap(photo, signed)), _bytes(overlay_heatmap(photo, signed, method="occlusion"))
    )
    # No negative value: Grad-CAM, smoothed.
    np.testing.assert_array_equal(
        _bytes(overlay_heatmap(photo, positive)), _bytes(overlay_heatmap(photo, positive, method="gradcam"))
    )
    # An explicit method still wins: a positive region map stays blocky when told so.
    blocky = _bytes(overlay_heatmap(np.zeros((8, 8, 3)), positive, method="lime", outline=False))
    smooth = _bytes(overlay_heatmap(np.zeros((8, 8, 3)), positive, outline=False))
    assert len(np.unique(blocky[..., 0])) < len(np.unique(smooth[..., 0]))


def test_gaussian_blur_matches_the_api_exactly():
    imaging = pytest.importorskip("kaimei_api.engine.imaging")
    rng = np.random.default_rng(7)
    for shape, sigma in [((37, 29, 3), 4.0), ((224, 224, 3), 16.0), ((9, 6, 3), 3.0), ((20, 30), 1.3)]:
        picture = rng.random(shape)
        ours, theirs = gaussian_blur(picture, sigma), imaging.gaussian_blur(picture, sigma)
        assert ours.shape == theirs.shape
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-12)


def test_gaussian_blur_keeps_flat_images_flat_and_mirrors_the_edges():
    flat = np.full((16, 20, 3), 0.37)
    np.testing.assert_allclose(gaussian_blur(flat, 5.0), flat, rtol=0, atol=1e-12)
    # σ ≤ 0 is the identity (a copy); any σ > 0 has a radius of at least ceil(3σ) = 1.
    picture = np.random.default_rng(0).random((6, 6, 3))
    assert not np.allclose(gaussian_blur(picture, 0.5), picture)
    for sigma in (0.0, -1.0):

        out = gaussian_blur(picture, sigma)
        np.testing.assert_array_equal(out, picture)
        assert out is not picture
    # One row, mirrored without repeating the edge: [a, b, c] with σ giving radius 1 → weights w0, w1, w0.
    row = np.asarray([[[0.0], [1.0], [0.0], [0.0]]])  # 1 × 4 × 1; radius is capped to the short side − 1 = 0
    np.testing.assert_array_equal(gaussian_blur(row, 2.0), row)
    tall = np.repeat(np.asarray([[0.0], [1.0], [0.0], [0.0]])[None, :, :], 4, axis=0)  # 4 × 4 × 1
    sigma = 0.5  # radius ceil(1.5) = 2
    taps = np.exp(-(np.arange(-2, 3) ** 2) / (2 * sigma**2))
    w = taps / taps.sum()
    # Column 0 mirrors to [0, 1 | 0, 1, 0, 0 | 0, 1]: reflect padding of [0, 1, 0, 0] by 2 is [0, 1, 0, 1, 0, 0, 0, 1]...
    padded = np.pad(np.asarray([0.0, 1.0, 0.0, 0.0]), 2, mode="reflect")
    expected = np.asarray([sum(w[k] * padded[i + k] for k in range(5)) for i in range(4)])
    np.testing.assert_allclose(gaussian_blur(tall, sigma)[0, :, 0], expected, atol=1e-12)


# --- explanation records ---------------------------------------------------------------------------


def test_focus_threshold_reads_the_record_or_falls_back_per_method():
    assert focus_threshold(_record("gradcam", focus={"threshold": 0.37})) == 0.37
    assert focus_threshold(_record("gradcam")) == 0.5
    assert focus_threshold(_record("occlusion", heat=[[0.8, -1.0], [0.2, 0.0]])) == pytest.approx(0.4)
    assert focus_threshold(_record("lime", heat=[[-0.8, -1.0]])) == 0.0
    # Options alone work too, with the method passed in.
    assert focus_threshold({"heatmap": [[0.6]]}, method="occlusion") == pytest.approx(0.3)
    assert focus_threshold({"heatmap": [[0.6]]}) == 0.5


def test_render_explanation_reads_method_heatmap_and_focus_from_the_record():
    photo = _photo(12, 12)
    record = _record("occlusion", heat=[[1.0, -0.4], [0.1, 0.05]], focus={"threshold": 0.9})
    for mode in ("heatmap", "spotlight", "original"):
        expected = overlay_heatmap(photo, record.options["heatmap"], method="occlusion", mode=mode, threshold=0.9)
        np.testing.assert_array_equal(_bytes(render_explanation(photo, record, mode=mode)), _bytes(expected))
    # Grad-CAM records are smoothed (bilinear), region records are not, even without negatives.
    smooth = render_explanation(photo, _record("gradcam", heat=[[1.0, 0.0], [0.0, 0.0]]), mode="original")
    assert smooth.size == (12, 12)
    gradcam = _bytes(render_explanation(np.zeros((8, 8, 3)), _record("gradcam", heat=[[1.0, 0.0], [0.0, 0.0]]), outline=False))
    blocky = _bytes(render_explanation(np.zeros((8, 8, 3)), _record("lime", heat=[[1.0, 0.0], [0.0, 0.0]]), outline=False))
    assert len(np.unique(gradcam[..., 0])) > len(np.unique(blocky[..., 0])) == 2
    # The feature's value_range maps 0–255 photos.
    feature = {"name": "photo", "type": "image", "value_range": [0, 255]}
    scaled = render_explanation(np.asarray(photo).astype(float), record, feature=feature)
    np.testing.assert_array_equal(_bytes(scaled), _bytes(render_explanation(photo, record)))
    with pytest.raises(ValueError, match="no heatmap"):
        render_explanation(photo, APIObject({"method": "shap", "options": {}}))


def test_evidence_images_gray_the_focus_or_everything_else():
    base = np.linspace(0.0, 0.9, 4 * 4 * 3).reshape(4, 4, 3)
    record = _record("occlusion", heat=[[1.0, 0.0], [0.0, 0.0]], focus={"threshold": 0.5})
    full, without, only = evidence_images(base, record)
    full, without, only = _bytes(full), _bytes(without), _bytes(only)
    np.testing.assert_array_equal(full, np.floor(base * 255 + 0.5).astype(int))
    assert (without[:2, :2] == 128).all() and (without[2:, :] == full[2:, :]).all()
    assert (only[:2, :2] == full[:2, :2]).all() and (only[2:, :] == 128).all() and (only[:2, 2:] == 128).all()
    mask = focus_mask(record, 4, 4)
    assert mask.sum() == 4 and mask[0, 0] and not mask[3, 3]
    # No positive heat → no focus region at all.
    assert not focus_mask(_record("lime", heat=[[-1.0, -0.5]]), 4, 4).any()
    # An explicit gray fill is the same as none.
    gray = _record("occlusion", heat=[[1.0, 0.0], [0.0, 0.0]], focus={"threshold": 0.5}, evidence={"fill": "gray"})
    np.testing.assert_array_equal(_bytes(evidence_images(base, gray)[1]), without)


def test_evidence_images_blur_the_hidden_part_when_the_api_did():
    rng = np.random.default_rng(3)
    base = rng.random((12, 10, 3))
    heat = np.zeros((6, 5))
    heat[1:3, 1:3] = 1.0
    record = _record("lime", heat=heat.tolist(), focus={"threshold": 0.5}, background="blur", blur_sigma=9.0,
                     evidence={"label": "cat", "full": 0.9, "without_focus": 0.2, "focus_only": 0.8, "fill": "blur",
                               "blur_sigma": 2.5})
    _, without, only = evidence_images(base, record)
    mask = focus_mask(record, 12, 10)[..., None]
    blurred = gaussian_blur(base, 2.5)  # the evidence's own sigma wins over options.blur_sigma
    np.testing.assert_array_equal(_bytes(without), np.floor(np.where(mask, blurred, base) * 255 + 0.5).astype(int))
    np.testing.assert_array_equal(_bytes(only), np.floor(np.where(mask, base, blurred) * 255 + 0.5).astype(int))
    # Without evidence.blur_sigma the record's blur_sigma is used.
    evidence = {k: v for k, v in record.to_dict()["options"]["evidence"].items() if k != "blur_sigma"}
    fallback = _record("lime", heat=heat.tolist(), focus={"threshold": 0.5}, blur_sigma=9.0, evidence=evidence)
    expected = np.floor(np.where(mask, gaussian_blur(base, 9.0), base) * 255 + 0.5).astype(int)
    np.testing.assert_array_equal(_bytes(evidence_images(base, fallback)[1]), expected)


def test_region_mask_uses_segments_or_rebuilds_the_grid():
    record = _record(
        "lime",
        heat=[[0.9, -0.2], [0.1, 0.0]],
        segments=[[0, 1], [2, 3]],
        regions=[{"id": 1, "name": "upper right", "attribution": -0.2}, {"id": 0, "name": "upper left", "attribution": 0.9}],
    )
    mask = region_mask(record, "upper right", 4, 6)
    assert mask.shape == (4, 6) and mask[:2, 3:].all() and mask.sum() == 6
    np.testing.assert_array_equal(region_mask(record, 1, 4, 6), mask)
    with pytest.raises(KeyError):
        region_mask(record, "the dog", 4, 6)
    # A record from before segments: grid patches named row/col (or top left … for 4 × 4).
    old = _record("occlusion", heat=[[0.0, 0.0], [1.0, 0.0]], grid=2)
    np.testing.assert_array_equal(region_mask(old, "row 1 col 0", 4, 4)[2:, :2], True)
    assert region_mask(old, "row 1 col 0", 4, 4).sum() == 4
    four = _record("occlusion", heat=np.zeros((4, 4)).tolist(), grid=4)
    assert region_mask(four, "bottom right", 8, 8)[6:, 6:].all() and region_mask(four, "bottom right", 8, 8).sum() == 4


def _regions(*values):
    return [
        {"id": i, "name": f"region {i}", "attribution": v, "anchor": [0.1 * i, 0.5], "centroid": [0.1 * i, 0.4]}
        for i, v in enumerate(values)
    ]


def test_numbered_regions_lists_support_then_opposition_with_running_numbers():
    record = _record("lime", regions=_regions(0.4, -2.0, 1.5, -0.1, 0.0, 0.9, -0.7, 0.2, 0.3, 0.6, -0.3, -0.05))
    items = numbered_regions(record)
    assert [(i["number"], i["group"], i["name"]) for i in items] == [
        (1, "for", "region 2"),
        (2, "for", "region 5"),
        (3, "for", "region 9"),
        (4, "for", "region 0"),
        (5, "for", "region 8"),
        (6, "against", "region 1"),
        (7, "against", "region 6"),
        (8, "against", "region 10"),
    ]
    assert items[0]["anchor"] == (0.2, 0.5) and items[0]["attribution"] == 1.5 and items[0]["id"] == 2
    # The caps are adjustable, the zero region is never listed, and Grad-CAM has no "against" group.
    assert [i["group"] for i in numbered_regions(record, for_count=2, against_count=1)] == ["for", "for", "against"]
    gradcam = _record("gradcam", regions=_regions(0.8, 0.7, 0.9))
    assert [i["name"] for i in numbered_regions(gradcam)] == ["region 2", "region 0", "region 1"]
    # Without an anchor the centroid is used; records without regions fall back to attributions.
    no_anchor = _record("occlusion", regions=[{"id": 0, "name": "centre", "attribution": 1.0, "centroid": [0.5, 0.6]}])
    assert numbered_regions(no_anchor)[0]["anchor"] == (0.5, 0.6)
    old = numbered_regions(_record("occlusion"))
    assert [(i["name"], i["group"], i["anchor"]) for i in old] == [
        ("upper left", "for", None),
        ("lower left", "for", None),
        ("lower right", "for", None),
        ("upper right", "against", None),
    ]


def test_explanation_summary_prefers_the_api_sentence_and_falls_back_to_the_heatmap():
    record = _record("gradcam", summary="The model looks mostly at the centre of the photo.")
    assert explanation_summary(record) == "The model looks mostly at the centre of the photo."
    heat = np.zeros((8, 8))
    heat[:3, 5:] = 1.0  # all the heat in the upper right corner
    fallback = explanation_summary(_record("gradcam", heat=heat.tolist()))
    assert "P(cat) = 98.0%" in fallback and "upper right" in fallback and "% of the photo" in fallback
    spread = explanation_summary(_record("gradcam", heat=np.ones((4, 4)).tolist()))
    assert "spread across 100% of the photo" in spread and "centre" in spread
    against = explanation_summary(_record("lime", heat=[[-1.0]]))
    assert against == "The model says cat with P(cat) = 98.0%. No part of the photo supports cat."


def test_regression_records_print_values_not_probabilities():
    output = {"label": None, "value": 12.3456, "name": "price"}
    record = _record("occlusion", heat=heat_corner(), output=output, score_space="value")
    text = explanation_summary(record)
    assert text.startswith("The model predicts 12.3. Half of the evidence for the prediction")
    assert "P(" not in text
    # No score_space (Grad-CAM) and no probabilities in the output: also a regression.
    gradcam = _record("gradcam", heat=[[-1.0]], output={"value": 0.004567})
    assert explanation_summary(gradcam) == "The model predicts 0.00457. No part of the photo pushes the prediction up."


def heat_corner():
    heat = np.zeros((4, 4))
    heat[0, 0] = 1.0
    return heat.tolist()


# --- plot_explanation ------------------------------------------------------------------------------


def _plot_record(**overrides):
    options = dict(
        heat=[[1.0, -0.4], [0.1, 0.05]],
        focus={"threshold": 0.6},
        evidence={"label": "cat", "full": 0.98, "without_focus": 0.03, "focus_only": 0.91, "fill": "blur", "blur_sigma": 2.0},
        background="blur",
        blur_sigma=2.0,
        regions=[
            {"id": 0, "name": "upper left", "attribution": 2.1, "anchor": [0.25, 0.25]},
            {"id": 1, "name": "upper right", "attribution": -0.7, "anchor": [0.75, 0.25]},
            {"id": 2, "name": "lower left", "attribution": 0.2, "anchor": [0.25, 0.75]},
            {"id": 3, "name": "lower right", "attribution": 0.0, "anchor": [0.75, 0.75]},
        ],
        segments=[[0, 1], [2, 3]],
        score_space="log_odds",
        summary="The model looks mostly at the upper left of the photo: half of its evidence for cat sits in 25% of the picture.",
    )
    options.update(overrides)
    method = options.pop("method", "occlusion")
    return _record(method, **options)


def _chart_axes(fig):
    """(label column, bars) of plot_explanation's region chart: the last two axes."""
    return fig.axes[-2], fig.axes[-1]


def test_plot_explanation_draws_photos_badges_evidence_and_two_region_groups():
    plt = pytest.importorskip("matplotlib.pyplot")
    photo = _photo(16, 16)
    fig = plot_explanation_or_skip(photo, _plot_record())
    try:
        assert len(fig.axes) == 3 + 3 + 2  # photo, heatmap, spotlight; three evidence pictures; the chart
        titles = [ax.get_title() for ax in fig.axes[:6]]
        assert titles[0] == "The model says cat\nP(cat) = 98.0%"
        assert titles[1].startswith("Red: for cat · blue: against") and "Spotlight" in titles[2]
        assert titles[3:] == ["Full photo\nP(cat) = 98.0%", "Focus blurred\nP(cat) = 3.0%", "Only the focus\nP(cat) = 91.0%"]
        # Badges on the heatmap, at the regions' anchors (fractions → pixel centres), in group colours.
        badges = {t.get_text(): t for t in fig.axes[1].texts}
        assert sorted(badges) == ["1", "2", "3"]
        assert badges["1"].get_position() == pytest.approx((0.25 * 16 - 0.5, 0.25 * 16 - 0.5), abs=0.6)
        assert badges["1"].get_bbox_patch().get_facecolor()[:3] == pytest.approx(_rgb(BAR_POSITIVE))
        assert badges["3"].get_bbox_patch().get_facecolor()[:3] == pytest.approx(_rgb(BAR_NEGATIVE))
        assert all(t.get_color() == "white" for t in badges.values())
        # The chart: "For cat" (strongest first), then "Against cat", numbered like the badges.
        labels, bars = _chart_axes(fig)
        texts = [t.get_text() for t in sorted(labels.texts, key=lambda t: (-t.get_position()[1], t.get_position()[0]))]
        assert texts == ["For cat", "1", "upper left", "2", "lower left", "Against cat", "3", "upper right"]
        values = sorted((t.get_position()[1], t.get_text()) for t in bars.texts)
        assert [text for _, text in values][::-1] == ["+2.10", "+0.20", "-0.70"]
        assert "against cat" in bars.get_xlabel() and "for cat" in bars.get_xlabel() and "log-odds" in bars.get_xlabel()
        assert fig._suptitle.get_text().startswith("The model looks mostly at the upper left")
        note = " ".join(t.get_text() for t in fig.texts)
        assert "blurred" in note and "grayed" not in note
        _assert_nothing_collides(fig)
    finally:
        plt.close("all")


def test_plot_explanation_grad_cam_lists_strongest_regions_and_gray_records_say_hidden():
    plt = pytest.importorskip("matplotlib.pyplot")
    record = _plot_record(
        method="gradcam",
        heat=[[1.0, 0.3], [0.5, 0.2]],
        evidence={"label": "cat", "full": 0.98, "without_focus": 0.5, "focus_only": 0.9, "fill": "gray"},
        background=None,
        regions=[{"id": i, "name": f"part {i}", "attribution": 0.9 - 0.05 * i, "anchor": [0.1 * i + 0.1, 0.5]} for i in range(7)],
        segments=[[0, 1, 2, 3], [4, 5, 6, 6]],
    )
    fig = plot_explanation_or_skip(_photo(16, 16), record)
    try:
        labels, _ = _chart_axes(fig)
        texts = [t.get_text() for t in labels.texts]
        assert texts[0] == "Strongest regions" and "Against cat" not in texts
        assert [t for t in texts if t.isdigit()] == ["1", "2", "3", "4", "5"]  # at most five
        assert fig.axes[4].get_title().startswith("Focus hidden")
        assert fig.axes[1].get_title().startswith("Where the model looks for cat")
        _assert_nothing_collides(fig)
    finally:
        plt.close("all")


def test_plot_explanation_formats_regression_values_with_three_significant_figures():
    plt = pytest.importorskip("matplotlib.pyplot")
    record = _plot_record(
        output={"value": 1234.5678, "name": "price"},
        score_space="value",
        evidence={"label": "price", "full": 1234.5678, "without_focus": 0.012345, "focus_only": 987.65, "fill": "gray"},
        summary=None,
    )
    fig = plot_explanation_or_skip(_photo(16, 16), record)
    try:
        titles = [ax.get_title() for ax in fig.axes[:6]]
        assert titles[0] == "The model predicts\n1.23e+03"
        assert titles[3:] == ["Full photo\nprediction = 1.23e+03", "Focus hidden\nprediction = 0.0123",
                              "Only the focus\nprediction = 988"]
        assert not any("P(" in t or "%" in t for t in titles)
        assert fig._suptitle.get_text().startswith("The model predicts 1.23e+03.")
        _, bars = _chart_axes(fig)
        assert "change in the prediction" in bars.get_xlabel() and "log-odds" not in bars.get_xlabel()
    finally:
        plt.close("all")


def test_plot_explanation_wraps_the_summary_to_stacked_subfigures():
    plt = pytest.importorskip("matplotlib.pyplot")
    long = " ".join(["The evidence for cat sits around the centre of the photo and the eyes."] * 5)
    outer = plt.figure(figsize=(13.6, 15.6))
    top, bottom = outer.subfigures(2, 1)
    try:
        for sub in (top, bottom):
            assert plot_explanation_or_skip(_photo(16, 16), _plot_record(summary=long), figure=sub) is sub
            assert len(sub.axes) == 8
            assert sub._suptitle.get_text().count("\n") >= 2  # wrapped, not one long line
        outer.canvas.draw()
        renderer = outer.canvas.get_renderer()
        for sub in (top, bottom):
            box = sub.bbox
            title = sub._suptitle.get_window_extent(renderer)
            assert box.x0 <= title.x0 and title.x1 <= box.x1 and box.y0 <= title.y0 and title.y1 <= box.y1
            for ax in sub.axes:
                extent = ax.get_window_extent(renderer)
                assert box.x0 - 1 <= extent.x0 and extent.x1 <= box.x1 + 1
                assert box.y0 - 1 <= extent.y0 and extent.y1 <= box.y1 + 1
            # The title stays above every panel.
            assert title.y0 >= max(ax.get_window_extent(renderer).y1 for ax in sub.axes)
        # Records with neither regions nor evidence still draw, into a narrow subfigure too.
        narrow = plt.figure(figsize=(8, 5)).subfigures(1, 1)
        plain = APIObject({**_record("gradcam", heat=[[1.0, 0.0], [0.0, 0.0]]).to_dict(), "attributions": {}})

        drawn = plot_explanation_or_skip(_photo(16, 16), plain, figure=narrow, modes=("heatmap",))
        assert "not measured" in drawn.axes[3].get_title() or "not measured" in drawn.axes[4].get_title()
        assert any("No regions" in t.get_text() for ax in drawn.axes for t in ax.texts)
    finally:
        plt.close("all")


def test_plot_explanation_keeps_the_predicted_class_in_the_photo_title():
    plt = pytest.importorskip("matplotlib.pyplot")
    why_dog = _record("gradcam")
    why_dog = APIObject(
        {
            **why_dog.to_dict(),
            "output": {**why_dog.to_dict()["output"], "target_label": "dog", "value": 0.02},
            "options": {**why_dog.to_dict()["options"], "target": "dog"},
        }
    )
    try:
        fig = plot_explanation_or_skip(_photo(16, 16), why_dog)
        assert fig.axes[0].get_title() == "The model says cat\nP(cat) = 98.0%"
        assert fig.axes[1].get_title().startswith("Where the model looks for dog")
        assert fig.axes[3].get_title() == "Full photo\nP(dog) = 2.0%"
    finally:
        plt.close("all")


def _rgb(colour: str):
    value = colour.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _assert_nothing_collides(fig) -> None:
    """Value labels stay inside the bars' axes and clear of the region names; texts stay on the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels, bars = _chart_axes(fig)
    bars_box = bars.get_window_extent(renderer)
    names = [t.get_window_extent(renderer) for t in labels.texts]
    for text in bars.texts:
        extent = text.get_window_extent(renderer)
        assert bars_box.x0 <= extent.x0 and extent.x1 <= bars_box.x1, text.get_text()
        assert all(not extent.overlaps(name) for name in names), text.get_text()
    figure_box = fig.bbox
    for text in [*fig.texts, fig._suptitle, *labels.texts, bars.xaxis.label]:
        extent = text.get_window_extent(renderer)
        assert figure_box.x0 <= extent.x0 and extent.x1 <= figure_box.x1 + 0.5, text.get_text()


def plot_explanation_or_skip(*args, **kwargs):
    import matplotlib

    from kaimei.images import plot_explanation

    matplotlib.use("Agg")
    return plot_explanation(*args, **kwargs)


# --- parity with the dashboard -------------------------------------------------------------------


def _load_generator():
    spec = importlib.util.spec_from_file_location("heatmap_render_fixture", FIXTURES / "heatmap_render.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_heatmap_render_fixture_matches_the_implementation():
    """``tests/fixtures/heatmap_render.json`` is what the dashboard checks itself against, so it
    must be exactly what this code renders; regenerate it with the script when the spec changes.
    Python reproduces every byte, the blurred evidence thumbnails included; only a JavaScript
    renderer gets the ±1 ``tolerance`` there (``Math.exp`` vs NumPy's ``exp``)."""
    generator = _load_generator()
    stored = json.loads((FIXTURES / "heatmap_render.json").read_text())
    assert (FIXTURES / "heatmap_render.json").stat().st_size < 60_000
    assert stored["format"] == "kaimei.heatmap_render.v1"
    assert [case["name"] for case in stored["cases"]] == [case["name"] for case in generator.CASES]
    blurred = 0
    for case in stored["cases"]:
        rendered = overlay_heatmap(
            case["base"], case["heat"], method=case["method"], mode=case["mode"], threshold=case["threshold"]
        )
        assert _bytes(rendered).ravel().tolist() == case["expected"], case["name"]
        options = {"heatmap": case["heat"]}
        if case["threshold"] is not None:
            options["focus"] = {"threshold": case["threshold"]}
        if case.get("evidence"):
            options["evidence"] = case["evidence"]
        _, without, only = evidence_images(case["base"], {"method": case["method"], "options": options})
        assert _bytes(without).ravel().tolist() == case["without_focus"], case["name"]
        assert _bytes(only).ravel().tolist() == case["focus_only"], case["name"]
        if (case.get("evidence") or {}).get("fill") == "blur":
            blurred += 1
            assert case["tolerance"] == {"expected": 0, "without_focus": 1, "focus_only": 1}
            assert case["without_focus"] != case["expected"]  # the blur really changed pixels
    assert blurred == 2
    assert generator.build()["cases"] == stored["cases"]


def test_palette_breakpoint_case_matches_hand_computed_bytes():
    stored = json.loads((FIXTURES / "heatmap_render.json").read_text())
    case = next(c for c in stored["cases"] if c["name"] == "lime_palette_breakpoints")
    pixels = np.asarray(case["expected"]).reshape(6, 6, 3)
    base = np.asarray(case["base"])

    def blend(y, x, colour, alpha):
        return np.floor((base[y, x] * (1 - alpha) + np.asarray(colour) * alpha) * 255 + 0.5).astype(int).tolist()

    # Rows 0–2 show the positive heat row, rows 3–5 the negative one; columns are 0, .12, .42, .5, .72, 1.
    # Pixels away from the focus outline (heat ≥ 0.45 in rows 0–2) carry only the palette.
    assert pixels[0, 0].tolist() == blend(0, 0, (0, 0, 0), 0.0)  # heat 0: the photo itself
    assert pixels[4, 1].tolist() == blend(4, 1, (0, 0, 0), 0.0)  # −0.12: still transparent
    assert pixels[1, 4].tolist() == blend(1, 4, (0.9692, 0.3652, 0.164), 0.66)  # +0.72
    assert pixels[4, 2].tolist() == blend(4, 2, (0.382, 0.482, 1.0), 0.33)  # −0.42
    r = (0.5 - 0.12) / 0.6
    assert pixels[4, 3].tolist() == blend(4, 3, (0.35, 0.45, 1.0), 0.66 * r * r * (3 - 2 * r))  # −0.5
    assert pixels[4, 4].tolist() == blend(4, 4, (0.328, 0.362, 0.978), 0.66)  # −0.72


# --- transport ---------------------------------------------------------------------------------------


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
