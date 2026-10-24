"""Photos in, heatmaps out: helpers for image models.

``load_image`` turns a photo into the nested ``height × width × channels`` list an image
feature reads, resized and center-cropped like torchvision's ``Resize`` + ``CenterCrop``::

    from kaimei.images import load_image, overlay_heatmap

    model = client.models.retrieve("cats-dogs-photos")
    photo = load_image("my_dog.jpg", model=model)
    explanation = client.explanations.create(model=model.slug, input={"photo": photo}, method="gradcam")
    overlay_heatmap(photo, explanation.options["heatmap"]).save("my_dog_gradcam.png")

Needs NumPy, plus Pillow for paths, bytes and PIL images: ``pip install "kaimei[images]"``.
"""

from __future__ import annotations

import io
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

Size = Union[int, Sequence[int]]

#: Overlay tints, as on the dashboard: cyan supports the explained class, pink works against it.
_SUPPORT = (0.243, 0.784, 1.0)
_AGAINST = (1.0, 0.42, 0.71)


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('Image helpers need NumPy: pip install "kaimei[images]"') from exc
    return numpy


def _pil() -> Any:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('Reading photos needs Pillow: pip install "kaimei[images]"') from exc
    return Image, ImageOps


def image_feature(model: Any) -> Mapping[str, Any]:
    """The image feature of a model record (or the feature itself, if given one)."""
    if isinstance(model, Mapping) and model.get("type") == "image":
        return model
    features = model.get("features") if isinstance(model, Mapping) else getattr(model, "features", None)
    for feature in features or []:
        if isinstance(feature, Mapping) and feature.get("type") == "image":
            return feature
    raise ValueError("This model has no image feature.")


def load_image(
    source: Any,
    size: Optional[Size] = None,
    *,
    model: Any = None,
    crop: str = "center",
    channels: Optional[int] = None,
    value_range: Optional[Sequence[float]] = None,
    decimals: int = 4,
) -> List[Any]:
    """A photo as the nested list an image feature reads.

    ``source`` is a file path, the file's bytes, a PIL image, or an array (``h × w × c`` or
    ``c × h × w``; integers are 0–255, floats are already ``[0, 1]``). ``model`` (a model record
    or its image feature) supplies the size, channels and ``value_range``; ``size`` (``224`` or
    ``(height, width)``) and the other arguments override it. ``crop="center"`` scales the
    shorter side to fit and crops the middle, so photos keep their proportions;
    ``crop="stretch"`` resizes to the exact size. Values are rounded to ``decimals`` places,
    which keeps the request small and is finer than 8-bit pixels.
    """
    np = _numpy()
    feature: Mapping[str, Any] = image_feature(model) if model is not None else {}
    height, width = _target_size(size, feature)
    depth = int(channels or feature.get("channels") or 3)
    low, high = (float(v) for v in (value_range or feature.get("value_range") or (0.0, 1.0)))
    if crop not in ("center", "stretch"):
        raise ValueError(f"crop must be 'center' or 'stretch', got {crop!r}.")
    pixels = _unit_pixels(source, height, width, depth, crop)
    return np.round(low + pixels * (high - low), decimals).tolist()


def overlay_heatmap(
    image: Any,
    heatmap: Any,
    *,
    value_range: Sequence[float] = (0.0, 1.0),
    opacity: float = 0.72,
) -> Any:
    """The explained image with its heatmap painted over it, as a PIL image.

    ``image`` is what you sent (a nested list or array in ``value_range`` units, or a PIL image);
    ``heatmap`` is ``explanation.options["heatmap"]``, values in ``[-1, 1]`` (``[0, 1]`` for
    Grad-CAM) on a grid that may be coarser than the image.
    """
    np = _numpy()
    Image, _ = _pil()
    if hasattr(image, "convert") and hasattr(image, "size"):
        base = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    else:
        low, high = (float(v) for v in value_range)
        base = (np.asarray(image, dtype=float) - low) / ((high - low) or 1.0)
        if base.ndim == 2:
            base = base[..., None]
        base = np.repeat(base[..., :1], 3, axis=-1) if base.shape[-1] == 1 else base[..., :3]
    base = np.clip(base, 0.0, 1.0)
    heat = np.clip(np.asarray(heatmap, dtype=float), -1.0, 1.0)
    if heat.ndim != 2:
        raise ValueError("heatmap must be a 2-D grid.")
    h, w = base.shape[:2]
    # Each pixel reads the heatmap cell it falls in, as the dashboard draws it.
    rows = np.minimum((np.arange(h) * heat.shape[0]) // h, heat.shape[0] - 1)
    cols = np.minimum((np.arange(w) * heat.shape[1]) // w, heat.shape[1] - 1)
    cells = heat[rows][:, cols]
    alpha = (np.abs(cells) * float(opacity))[..., None]
    tint = np.where(cells[..., None] >= 0, np.asarray(_SUPPORT), np.asarray(_AGAINST))
    blended = np.clip(base * (1 - alpha) + tint * alpha, 0.0, 1.0)
    return Image.fromarray(np.round(blended * 255).astype(np.uint8))


# --- internals -------------------------------------------------------------------------------


def _target_size(size: Optional[Size], feature: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    if size is None:
        height, width = feature.get("height"), feature.get("width")
        return (int(height) if height else None), (int(width) if width else None)
    if isinstance(size, (int, float)):
        return int(size), int(size)
    items = [int(v) for v in size]
    if len(items) != 2:
        raise ValueError(f"size must be one number or (height, width), got {list(size)}.")
    return items[0], items[1]


def _unit_pixels(source: Any, height: Optional[int], width: Optional[int], depth: int, crop: str) -> Any:
    """``(height, width, depth)`` floats in ``[0, 1]``."""
    np = _numpy()
    image = _as_pil(source)
    if image is None:
        array = _array_pixels(source)
        if (height is None or array.shape[0] == height) and (width is None or array.shape[1] == width):
            return _with_channels(array, depth)
        Image, _ = _pil()
        if array.shape[-1] == 1:
            image = Image.fromarray(np.round(array[..., 0] * 255).astype(np.uint8))
        else:
            image = Image.fromarray(np.round(array[..., :3] * 255).astype(np.uint8))
    image = image.convert("L" if depth == 1 else "RGB")
    image = _resize(image, height or image.height, width or image.width, crop)
    array = np.asarray(image, dtype=float) / 255.0
    return _with_channels(array[..., None] if array.ndim == 2 else array, depth)


def _as_pil(source: Any) -> Any:
    if hasattr(source, "convert") and hasattr(source, "size") and hasattr(source, "resize"):
        image = source
    elif isinstance(source, (str, os.PathLike)):
        Image, _ = _pil()
        with Image.open(source) as opened:
            opened.load()
            image = opened.copy()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        Image, _ = _pil()
        image = Image.open(io.BytesIO(bytes(source)))
        image.load()
    else:
        return None
    _, ImageOps = _pil()
    # Phone photos store their rotation in EXIF; browsers apply it, so do the same here.
    return ImageOps.exif_transpose(image)


def _array_pixels(source: Any) -> Any:
    np = _numpy()
    array = np.asarray(source)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"An image array must be h × w × c or c × h × w, got shape {array.shape}.")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(float) / 255.0
    return np.clip(array.astype(float), 0.0, 1.0)


def _with_channels(array: Any, depth: int) -> Any:
    np = _numpy()
    have = array.shape[-1]
    if have == depth:
        return array
    if depth == 1:
        # ITU-R 601 luma, as PIL's "L" conversion.
        weights = np.asarray([0.299, 0.587, 0.114])
        return (array[..., :3] @ weights)[..., None] if have >= 3 else array[..., :1]
    if have == 1:
        return np.repeat(array, depth, axis=-1)
    if have > depth:
        return array[..., :depth]
    raise ValueError(f"Cannot turn a {have}-channel image into {depth} channels.")


def _resize(image: Any, height: int, width: int, crop: str) -> Any:
    Image, _ = _pil()
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    if crop == "stretch" or (image.width, image.height) == (width, height):
        return image if (image.width, image.height) == (width, height) else image.resize((width, height), bilinear)
    # Scale the shorter side to fit and crop the middle: torchvision's Resize(n) + CenterCrop(n).
    if width / image.width >= height / image.height:
        new_w, new_h = width, max(height, int(width * image.height / image.width))
    else:
        new_w, new_h = max(width, int(height * image.width / image.height)), height
    if (new_w, new_h) != (image.width, image.height):
        image = image.resize((new_w, new_h), bilinear)
    top = int(round((new_h - height) / 2.0))
    left = int(round((new_w - width) / 2.0))
    return image.crop((left, top, left + width, top + height))
