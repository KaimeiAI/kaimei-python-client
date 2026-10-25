"""Photos in, heatmaps out: helpers for image models.

``load_image`` turns a photo into the nested ``height × width × channels`` list an image
feature reads, resized and center-cropped like torchvision's ``Resize`` + ``CenterCrop``.
The rendering helpers draw an explanation exactly as the dashboard's Explain tab does
(same colours, same focus outline, byte for byte)::

    from kaimei.images import explanation_summary, load_image, plot_explanation, render_explanation

    model = client.models.retrieve("cats-dogs-photos")
    photo = load_image("my_dog.jpg", model=model)
    explanation = client.explanations.create(model=model.slug, input={"photo": photo}, method="gradcam")
    print(explanation_summary(explanation))                      # where the model looks, in words
    render_explanation(photo, explanation).save("my_dog_gradcam.png")          # warm glow on the evidence
    render_explanation(photo, explanation, mode="spotlight").save("my_dog_spot.png")
    plot_explanation(photo, explanation)                         # the full dashboard view (matplotlib)

Colours follow the SHAP convention: amber → orange → red where the photo is evidence *for* the
explained class, light blue → indigo where it is evidence *against* it, never more than 66%
opaque, so the eyes and nose under the hot spot stay visible.

Rendering pieces, for custom views: ``upsample_heatmap``, ``heat_color`` / ``heat_alpha``,
``focus_threshold``, ``overlay_heatmap``, ``focus_mask``, ``region_mask``, ``gaussian_blur``,
``evidence_images`` and ``numbered_regions``.

Needs NumPy, plus Pillow for paths, bytes and PIL images: ``pip install "kaimei[images]"``.
``plot_explanation`` also needs matplotlib, which stays optional. Importing this module loads
neither Pillow nor matplotlib.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

Size = Union[int, Sequence[int]]

__all__ = [
    "image_feature",
    "load_image",
    "upsample_heatmap",
    "heat_color",
    "heat_alpha",
    "focus_threshold",
    "overlay_heatmap",
    "render_explanation",
    "focus_mask",
    "region_mask",
    "gaussian_blur",
    "evidence_images",
    "numbered_regions",
    "explanation_summary",
    "plot_explanation",
    "BAR_POSITIVE",
    "BAR_NEGATIVE",
    "SUPPORT_GRADIENT",
    "AGAINST_GRADIENT",
]

#: Heat colours at intensity 0, 0.5 and 1, shared with the dashboard. Evidence *for* the explained
#: class warms from amber through orange to red; evidence *against* it deepens from light blue to
#: indigo (the SHAP convention). Low heat stays transparent (see ``heat_alpha``), so only the
#: region the model really uses lights up.
_SUPPORT_STOPS = ((1.0, 0.86, 0.30), (1.0, 0.55, 0.12), (0.93, 0.13, 0.22))
_AGAINST_STOPS = ((0.55, 0.65, 1.0), (0.35, 0.45, 1.0), (0.30, 0.25, 0.95))
#: Alpha ramp: nothing up to 0.12, a smoothstep up to 0.66 at 0.72 and above, so the photo always
#: shows through the hottest spot.
_ALPHA_FLOOR, _ALPHA_SPAN, _ALPHA_MAX = 0.12, 0.60, 0.66
#: Spotlight mode dims everything outside the focus to this fraction, fading in over 0.12 of heat.
_SPOT_DIM, _SPOT_FADE = 0.28, 0.12
#: Focus outline: blend the edge pixels this far towards white.
_OUTLINE_MIX = 0.85
#: Gray fill of the evidence check, in [0, 1] pixel units (the API grays with the same value).
_EVIDENCE_GRAY = 0.5
#: The API's blur for hiding regions of a photo: σ = 16 px at a 224-pixel short side, scaled.
_BLUR_SIGMA_224 = 16.0

#: Solid colours for chart bars, legend chips and number badges that must match the picture:
#: orange for evidence for the explained class, indigo-blue for evidence against it.
BAR_POSITIVE = "#f97316"
BAR_NEGATIVE = "#5b5bf5"
#: The same two colours as weak → strong gradients (bars shaded by their size).
SUPPORT_GRADIENT = ("#ffb04d", "#ed2138")
AGAINST_GRADIENT = ("#8ca6ff", "#4c40f2")

_POSITION_ROWS = ("upper", "middle", "lower")
_POSITION_COLS = ("left", "centre", "right")
#: Grid-patch names the API uses for a 4×4 grid (older records; superpixel records carry names).
_PATCH_ROWS = ("top", "upper", "lower", "bottom")
_PATCH_COLS = ("left", "mid-left", "mid-right", "right")


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


# --- rendering ---------------------------------------------------------------------------------


def upsample_heatmap(heat: Any, height: int, width: int, kind: str = "bilinear") -> Any:
    """``heat`` (``mh × mw``) stretched to ``height × width`` floats, clamped to ``[-1, 1]``.

    ``kind="bilinear"`` is for continuous maps (Grad-CAM) and samples with half-pixel centres,
    like ``align_corners=False``; ``kind="nearest"`` keeps region maps (occlusion, LIME) blocky
    so every pixel shows its own region's value. Both match the dashboard pixel for pixel.
    """
    np = _numpy()
    grid = np.clip(np.asarray(heat, dtype=float), -1.0, 1.0)
    if grid.ndim != 2:
        raise ValueError("heatmap must be a 2-D grid.")
    if kind == "nearest":
        rows, cols = _nearest_index(grid.shape[0], height), _nearest_index(grid.shape[1], width)
        return grid[rows][:, cols]
    if kind != "bilinear":
        raise ValueError(f"kind must be 'bilinear' or 'nearest', got {kind!r}.")
    lo_r, hi_r, f_r = _bilinear_index(grid.shape[0], height)
    lo_c, hi_c, f_c = _bilinear_index(grid.shape[1], width)
    # Separable: rows first, then columns, in the dashboard's order so the rounding agrees.
    rows = grid[lo_r] * (1.0 - f_r)[:, None] + grid[hi_r] * f_r[:, None]
    out = rows[:, lo_c] * (1.0 - f_c)[None, :] + rows[:, hi_c] * f_c[None, :]
    return np.clip(out, -1.0, 1.0)


def heat_color(value: Any) -> Any:
    """The RGB colour (``… × 3`` floats in ``[0, 1]``) for heat ``value`` (any array shape).

    Intensity is ``|value|``; the sign picks the palette: ``>= 0`` (evidence for the explained
    class) runs amber → orange → red, ``< 0`` (against it) light blue → blue → indigo.
    """
    np = _numpy()
    v = np.asarray(value, dtype=float)
    a = np.abs(v)
    support = _ramp(a, _SUPPORT_STOPS)
    against = _ramp(a, _AGAINST_STOPS)
    return np.where((v >= 0)[..., None], support, against)


def heat_alpha(value: Any) -> Any:
    """Overlay opacity for heat ``value``: 0 up to 0.12, a smoothstep to 0.66 at 0.72 and above."""
    np = _numpy()
    r = np.clip((np.abs(np.asarray(value, dtype=float)) - _ALPHA_FLOOR) / _ALPHA_SPAN, 0.0, 1.0)
    return _ALPHA_MAX * r * r * (3.0 - 2.0 * r)


def focus_threshold(explanation: Any, method: Optional[str] = None) -> float:
    """The heat level that bounds the focus region (the white outline / spotlight).

    Reads ``options["focus"]["threshold"]`` when the API computed it (the smallest region holding
    half of the positive heat). Older records fall back to ``0.5`` for Grad-CAM and half of the
    largest positive value for signed maps. Accepts an explanation record or its ``options``.
    """
    np = _numpy()
    options = explanation.get("options") if isinstance(explanation, Mapping) and "options" in explanation else explanation
    options = options or {}
    focus = options.get("focus") if isinstance(options, Mapping) else None
    if isinstance(focus, Mapping) and focus.get("threshold") is not None:
        return float(focus["threshold"])
    method = method or (explanation.get("method") if isinstance(explanation, Mapping) else None) or "gradcam"
    if method == "gradcam":
        return 0.5
    heat = options.get("heatmap") if isinstance(options, Mapping) else None
    if heat is None:
        return 0.5
    return 0.5 * max(float(np.max(np.asarray(heat, dtype=float))), 0.0)


def overlay_heatmap(
    image: Any,
    heatmap: Any,
    *,
    value_range: Sequence[float] = (0.0, 1.0),
    method: Optional[str] = None,
    mode: str = "heatmap",
    threshold: Optional[float] = None,
    outline: bool = True,
    opacity: Optional[float] = None,
) -> Any:
    """The explained image with its heatmap drawn over it, as a PIL image.

    ``image`` is what you sent (a nested list or array in ``value_range`` units, or a PIL image);
    ``heatmap`` is ``explanation.options["heatmap"]``, values in ``[-1, 1]`` (``[0, 1]`` for
    Grad-CAM) on a grid that may be coarser than the image.

    ``method`` picks the upsampling: ``"gradcam"`` smooths the map (bilinear), any other method
    (``"occlusion"``, ``"lime"``) keeps one flat value per region (nearest). The default,
    ``None``, infers it from the values: a map with any negative value is a signed region map,
    anything else is treated as Grad-CAM. Pass ``method`` explicitly for a region map without
    negative values (or use :func:`render_explanation`, which reads it from the record).

    ``mode`` is ``"heatmap"`` (a warm glow whose colour and opacity grow with the heat, blue
    where the heat is negative), ``"spotlight"`` (the photo itself, dimmed outside the focus
    region) or ``"original"``. ``threshold`` bounds the focus region, by default
    :func:`focus_threshold`'s fallback for the method; pass the explanation's value to match the
    dashboard exactly. ``outline`` draws the focus region's white edge. ``opacity`` is accepted
    for calls written against 0.4 and ignored: the ramp is fixed so every surface agrees.
    """
    np = _numpy()
    base = _base_pixels(image, value_range)
    heat = np.clip(np.asarray(heatmap, dtype=float), -1.0, 1.0)
    if heat.ndim != 2:
        raise ValueError("heatmap must be a 2-D grid.")
    if method is None:
        method = "occlusion" if bool((heat < 0).any()) else "gradcam"
    if threshold is None:
        threshold = 0.5 if method == "gradcam" else 0.5 * max(float(heat.max()), 0.0)
    field = upsample_heatmap(heat, base.shape[0], base.shape[1], _upsample_kind(method))
    return _to_image(_compose(base, field, mode, float(threshold), outline))


def render_explanation(
    image: Any,
    explanation: Any,
    *,
    feature: Any = None,
    mode: str = "heatmap",
    threshold: Optional[float] = None,
    outline: bool = True,
) -> Any:
    """:func:`overlay_heatmap` driven by an explanation record (method, heatmap, focus).

    ``feature`` (the model record or its image feature) supplies ``value_range`` when the
    photo is not in ``[0, 1]`` units.
    """
    options = _options(explanation)
    if options.get("heatmap") is None:
        raise ValueError("This explanation has no heatmap; only image explanations can be rendered.")
    return overlay_heatmap(
        image,
        options["heatmap"],
        value_range=_value_range(feature),
        method=str(explanation.get("method") or "gradcam"),
        mode=mode,
        threshold=focus_threshold(explanation) if threshold is None else threshold,
        outline=outline,
    )


def focus_mask(explanation: Any, height: int, width: int, *, threshold: Optional[float] = None) -> Any:
    """``height × width`` booleans: the focus region (heat at or above the focus threshold)."""
    np = _numpy()
    options = _options(explanation)
    t = focus_threshold(explanation) if threshold is None else float(threshold)
    if t <= 0:
        # No positive heat at all: nothing to outline, nothing to spotlight.
        return np.zeros((height, width), dtype=bool)
    field = upsample_heatmap(options["heatmap"], height, width, _upsample_kind(explanation.get("method")))
    return field >= t


def region_mask(explanation: Any, region: Union[str, int], height: int, width: int) -> Any:
    """``height × width`` booleans covering one region, by name (an ``attributions`` key) or id.

    Reads ``options["segments"]`` and ``options["regions"]``; older grid records are rebuilt from
    ``options["grid"]`` and the ``row r col c`` / ``top left`` patch names.
    """
    np = _numpy()
    options = _options(explanation)
    segments = options.get("segments")
    if segments is not None:
        ids = np.asarray(segments, dtype=int)
        target = _region_id(options, region)
    else:
        ids, target = _grid_segments(options, region)
    rows, cols = _nearest_index(ids.shape[0], height), _nearest_index(ids.shape[1], width)
    return ids[rows][:, cols] == target


def gaussian_blur(image: Any, sigma: float) -> Any:
    """``image`` (``h × w × c`` floats, or ``h × w``) blurred with a separable Gaussian, edges mirrored.

    The same arithmetic as the API (``kaimei_api.engine.imaging.gaussian_blur``), which hides
    regions of a photo by replacing them with this blurred copy: radius ``ceil(3σ)`` (at most
    the short side − 1), weights ``exp(−k² / 2σ²)`` divided by their sequential sum, NumPy
    ``reflect`` padding (mirrored without repeating the edge pixel), a horizontal pass then a
    vertical pass, each accumulating its taps in order, all in float64. Returns ``h × w × c``.
    """
    np = _numpy()
    picture = np.asarray(image, dtype=float)
    if picture.ndim == 2:
        picture = picture[..., None]
    height, width = picture.shape[:2]
    radius = min(int(math.ceil(3.0 * float(sigma))), height - 1, width - 1)
    if radius < 1 or not sigma > 0:
        return picture.copy()
    taps = [math.exp(-(k * k) / (2.0 * float(sigma) ** 2)) for k in range(-radius, radius + 1)]
    total = 0.0
    for tap in taps:
        total += tap
    weights = [tap / total for tap in taps]
    padded = np.pad(picture, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    across = np.zeros_like(picture)
    for k, weight in enumerate(weights):
        across += weight * padded[:, k : k + width]
    padded = np.pad(across, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    out = np.zeros_like(picture)
    for k, weight in enumerate(weights):
        out += weight * padded[k : k + height]
    return out


def evidence_images(
    image: Any,
    explanation: Any,
    *,
    feature: Any = None,
    threshold: Optional[float] = None,
) -> Tuple[Any, Any, Any]:
    """``(full, without_focus, focus_only)`` PIL images: the photo, the photo with its focus
    region hidden, and the photo with everything else hidden — the three pictures the API scored
    for ``options["evidence"]``.

    Hidden pixels are replaced the way the API replaced them: with a blurred copy of the photo
    when ``options["evidence"]["fill"] == "blur"`` (σ from ``evidence["blur_sigma"]``, else
    ``options["blur_sigma"]``; see :func:`gaussian_blur`), with flat gray otherwise.
    """
    np = _numpy()
    base = _base_pixels(image, _value_range(feature))
    mask = focus_mask(explanation, base.shape[0], base.shape[1], threshold=threshold)[..., None]
    fill = _evidence_fill(_options(explanation), base)
    without = np.where(mask, fill, base)
    only = np.where(mask, base, fill)
    return _to_image(base), _to_image(without), _to_image(only)


def numbered_regions(explanation: Any, *, for_count: int = 5, against_count: int = 3) -> List[Dict[str, Any]]:
    """The regions worth pointing at, numbered the way the dashboard numbers them.

    First up to ``for_count`` regions with a positive attribution (evidence for the explained
    class), strongest first; then up to ``against_count`` with a negative one, most negative
    first (never any for Grad-CAM, whose values are all ≥ 0). Numbers run 1..N across both
    groups. Each item is ``{"number", "group" ("for" | "against"), "id", "name", "attribution",
    "anchor"}``, where ``anchor`` is ``(x, y)`` as fractions of the photo — a point deep inside
    the region for its number badge (the region's centroid for records without one; ``None``
    for records without regions, which only carry ``attributions``).
    """
    options = _options(explanation)
    regions = options.get("regions")
    items: List[Dict[str, Any]] = []
    if regions:
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            point = region.get("anchor") or region.get("centroid")
            items.append(
                {
                    "id": region.get("id"),
                    "name": str(region.get("name")),
                    "attribution": float(region.get("attribution") or 0.0),
                    "anchor": (float(point[0]), float(point[1])) if point is not None else None,
                }
            )
    else:
        for name, value in (explanation.get("attributions") or {}).items():
            items.append({"id": None, "name": str(name), "attribution": float(value), "anchor": None})
    # Stable sorts: equal values keep the API's order (|attribution| descending, then region id).
    support = sorted((i for i in items if i["attribution"] > 0), key=lambda i: -i["attribution"])
    against = sorted((i for i in items if i["attribution"] < 0), key=lambda i: i["attribution"])
    if (explanation.get("method") or "gradcam") == "gradcam":
        against = []
    chosen = [("for", i) for i in support[: max(int(for_count), 0)]]
    chosen += [("against", i) for i in against[: max(int(against_count), 0)]]
    return [{"number": n, "group": group, **item} for n, (group, item) in enumerate(chosen, 1)]


def explanation_summary(explanation: Any) -> str:
    """The plain-English summary of an image explanation.

    The API writes it as ``options["summary"]``; records from before 0.5 get one built here
    from the heatmap (where half of the positive heat sits) and the prediction. Regression
    records speak of "the prediction" and print values with three significant figures.
    """
    options = _options(explanation)
    summary = options.get("summary")
    if summary:
        return str(summary)
    np = _numpy()
    output = explanation.get("output") or {}
    regression = _is_regression(explanation)
    if regression:
        label = "the prediction"
    else:
        label = options.get("target") or output.get("target_label") or output.get("label") or "the prediction"
    heat = options.get("heatmap")
    sentence = ""
    if heat is not None:
        positive = np.maximum(np.asarray(heat, dtype=float), 0.0)
        total = float(positive.sum())
        if total > 0:
            # Spec A.3 on the stored grid: the smallest set of cells holding half of the heat.
            flat = np.sort(positive.ravel())[::-1]
            taken = int(np.searchsorted(np.cumsum(flat), 0.5 * total, side="left")) + 1
            region = positive >= flat[min(taken, flat.size) - 1]
            ys, xs = np.nonzero(region)
            area = region.mean() * 100.0
            position = _position_word((xs.mean() + 0.5) / region.shape[1], (ys.mean() + 0.5) / region.shape[0])
            if area >= 35:
                sentence = f"The evidence for {label} is spread across {area:.0f}% of the photo, centred on the {position}."
            else:
                sentence = f"Half of the evidence for {label} sits in {area:.0f}% of the photo, around the {position}."
    if not sentence:
        sentence = "No part of the photo pushes the prediction up." if regression else f"No part of the photo supports {label}."
    value = _target_probability(explanation)
    if value is not None:
        if regression:
            sentence = f"The model predicts {_format_value(value)}. {sentence}"
        else:
            sentence = f"The model says {output.get('label', label)} with P({label}) = {_percent(value)}. {sentence}"
    return sentence


def plot_explanation(
    image: Any,
    explanation: Any,
    *,
    feature: Any = None,
    modes: Sequence[str] = ("heatmap", "spotlight"),
    figure: Any = None,
    title: Optional[str] = None,
    for_regions: int = 5,
    against_regions: int = 3,
) -> Any:
    """The dashboard's explanation view as one matplotlib figure.

    Top row: the photo with the model's answer, then one panel per entry of ``modes`` — by
    default the heatmap, with the numbered regions drawn as badges on the picture, and the
    spotlight on the focus region. Bottom row: the evidence check — the full photo, the photo
    with its focus region blurred (or grayed, as the API did) and only the focus region, each
    with the model's probability. On the right, spanning both rows, the same numbers as a bar
    chart: up to ``for_regions`` regions "For {target}" (orange) and ``against_regions``
    "Against {target}" (blue); Grad-CAM lists its "Strongest regions" only. The summary
    sentence (or ``title``) is the figure title, wrapped to the figure's width.

    Pass ``figure`` (a Figure or SubFigure) to draw into an existing one, for example two
    explanations stacked; the panels shrink to fit it. Returns the figure. Needs matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot_explanation needs matplotlib: pip install matplotlib") from exc

    options = _options(explanation)
    output = explanation.get("output") or {}
    method = str(explanation.get("method") or "gradcam")
    target = str(options.get("target") or output.get("target_label") or output.get("label") or "target")
    regression = _is_regression(explanation)
    modes = [str(m) for m in modes]
    base = _base_pixels(image, _value_range(feature))

    fig = figure if figure is not None else plt.figure(figsize=(_PLOT_WIDTH, 8.0))
    width, height = _size_inches(fig)
    lines = _wrap_lines(title or explanation_summary(explanation), width - 2 * _PLOT_SIDE, _TITLE_SIZE)
    columns = max(len(modes) + 1, 3)
    if figure is None:
        height = _plot_height(width, columns, len(lines))
        fig.set_size_inches(width, height)
    box = _PlotBox(width, height, columns, len(lines))
    fig.suptitle(
        "\n".join(lines),
        x=_PLOT_SIDE / width,
        y=1.0 - _PLOT_TITLE_PAD / height,
        ha="left",
        va="top",
        fontsize=_TITLE_SIZE,
        linespacing=1.25,
    )

    # Top row: what the model says, then the requested views.
    items = numbered_regions(explanation, for_count=for_regions, against_count=against_regions)
    predicted = str(output.get("label", target))
    if regression:
        value = output.get("value", _target_probability(explanation))
        answer = f"The model predicts\n{'—' if value is None else _format_value(float(value))}"
    else:
        probabilities = output.get("probabilities") or {}
        p_predicted = probabilities.get(predicted) if isinstance(probabilities, Mapping) else None
        if p_predicted is None:
            p_predicted = output.get("probability", _target_probability(explanation))
        # The predicted class with its own probability: on "why dog?" for a photo the model calls
        # a cat, P(dog) printed next to the word "cat" would read as a contradiction.
        answer = f"The model says {predicted}\nP({predicted}) = {_percent(float(p_predicted or 0.0))}"
    subject = "the prediction" if regression else target
    looks = f"Red: for {subject} · blue: against" if method != "gradcam" else f"Where the model looks for {subject}"
    titles = {
        "heatmap": f"{looks}\nnumbers mark the regions in the chart" if items else looks,
        "spotlight": "Spotlight on the focus\nthe outline holds half of the evidence",
        "original": "The photo",
    }
    panels = [("original", answer)] + [(mode, titles.get(mode, mode)) for mode in modes]
    badge_panel = next((i for i, (mode, _) in enumerate(panels) if mode == "heatmap"), 0)
    for index, (mode, caption) in enumerate(panels):
        ax = fig.add_axes(box.panel(0, index))
        ax.imshow(render_explanation(image, explanation, feature=feature, mode=mode))
        ax.set_title(caption, fontsize=_CAPTION_SIZE * box.font, linespacing=1.3)
        ax.set_axis_off()
        if index == badge_panel and items:
            _draw_badges(ax, explanation, items, base.shape[0], base.shape[1], box)

    # Bottom row: the evidence check.
    evidence = options.get("evidence") or {}
    label = str(evidence.get("label") or target)
    hidden = "Focus blurred" if evidence.get("fill") == "blur" else "Focus hidden"
    captions = [
        ("Full photo", evidence.get("full", _target_probability(explanation))),
        (hidden, evidence.get("without_focus")),
        ("Only the focus", evidence.get("focus_only")),
    ]
    pictures = evidence_images(image, explanation, feature=feature)
    for index, (picture, (name, value)) in enumerate(zip(pictures, captions)):
        ax = fig.add_axes(box.panel(1, index))
        ax.imshow(picture)
        if value is None:
            text = "not measured"
        elif regression:
            text = f"prediction = {_format_value(float(value))}"
        else:
            text = f"P({label}) = {_percent(float(value))}"
        ax.set_title(f"{name}\n{text}", fontsize=_CAPTION_SIZE * box.font, linespacing=1.3)
        ax.set_axis_off()

    _draw_region_chart(fig, box, explanation, items, subject)
    return fig


# --- plot_explanation internals ------------------------------------------------------------------

#: plot_explanation's layout, in inches: the default width, the margins, the gap between photo
#: panels, the gap before the region chart, the chart's label column (badge + region name) and
#: bar column, one line of the summary title, the padding around it, the room for two-line panel
#: captions, and the bottom margin. Panels are square; their side follows from the width.
_PLOT_WIDTH = 13.6
_PLOT_SIDE, _PLOT_GAP, _PLOT_CHART_GAP = 0.12, 0.12, 0.42
_PLOT_LABELS, _PLOT_BARS = 1.55, 2.45
_PLOT_TITLE_LINE, _PLOT_TITLE_PAD, _PLOT_CAPTION, _PLOT_BOTTOM = 0.245, 0.14, 0.5, 0.12
#: Font sizes (points): the summary, panel captions, chart labels, badge numerals.
_TITLE_SIZE, _CAPTION_SIZE, _LABEL_SIZE, _BADGE_SIZE = 12.5, 10.0, 9.5, 9.5
#: Badge circle diameter in inches at _BADGE_SIZE (keeps badges on the picture from overlapping).
_BADGE_INCHES = 0.23
#: Header colours of the chart groups: darker shades of the badge colours, for contrast on white.
_FOR_INK, _AGAINST_INK = "#c2410c", "#4338ca"
#: Secondary text (region names, notes) and value labels.
_MUTED_INK, _VALUE_INK = "#57606a", "#3d434b"


class _PlotBox:
    """Where everything goes, in inches, for a (sub)figure of ``width × height`` inches."""

    def __init__(self, width: float, height: float, columns: int, title_lines: int) -> None:
        self.width, self.height = width, height
        chart = _PLOT_CHART_GAP + _PLOT_LABELS + _PLOT_BARS
        fixed = 2 * _PLOT_SIDE + (columns - 1) * _PLOT_GAP
        # A narrow (sub)figure squeezes the chart columns too, never below a third of their size.
        self.scale = max(min(1.0, (width - fixed - columns * 1.0) / chart), 0.33)
        self.labels, self.bars = _PLOT_LABELS * self.scale, _PLOT_BARS * self.scale
        self.top = height - (2 * _PLOT_TITLE_PAD + title_lines * _PLOT_TITLE_LINE + _PLOT_CAPTION)
        by_width = (width - fixed - chart * self.scale) / columns
        by_height = (self.top - _PLOT_CAPTION - _PLOT_BOTTOM) / 2
        self.panel_inches = max(min(by_width, by_height), 0.3)
        #: Text and badges shrink with panels smaller than the default (~2.9 inches), so a cramped
        #: (sub)figure stays legible instead of overlapping; never below 55%.
        self.font = min(1.0, max(0.55, self.panel_inches / 2.6))
        # When the height limits the panels, centre the whole layout horizontally.
        self.left = _PLOT_SIDE + max(by_width - self.panel_inches, 0.0) * columns / 2
        side = self.panel_inches
        self.chart_left = self.left + columns * side + (columns - 1) * _PLOT_GAP + _PLOT_CHART_GAP * self.scale
        self.chart_span = 2 * side + _PLOT_CAPTION  # from the top row's top to the bottom row's bottom

    def rect(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Inches (from the bottom left) to (sub)figure fractions, as ``Figure.add_axes`` takes them."""
        return (x / self.width, y / self.height, w / self.width, h / self.height)

    def panel(self, row: int, column: int) -> Tuple[float, float, float, float]:
        side = self.panel_inches
        y = self.top - side - row * (side + _PLOT_CAPTION)
        return self.rect(self.left + column * (side + _PLOT_GAP), y, side, side)


def _plot_height(width: float, columns: int, title_lines: int) -> float:
    """The figure height whose rows are exactly one square panel tall, so no blank bands appear."""
    chart = _PLOT_CHART_GAP + _PLOT_LABELS + _PLOT_BARS
    side = (width - 2 * _PLOT_SIDE - (columns - 1) * _PLOT_GAP - chart) / columns
    top = 2 * _PLOT_TITLE_PAD + title_lines * _PLOT_TITLE_LINE + _PLOT_CAPTION
    return top + 2 * side + _PLOT_CAPTION + _PLOT_BOTTOM


def _size_inches(fig: Any) -> Tuple[float, float]:
    if hasattr(fig, "get_size_inches"):
        width, height = fig.get_size_inches()
        return float(width), float(height)
    # A SubFigure has no size of its own: measure its box on the parent figure.
    return float(fig.bbox.width / fig.dpi), float(fig.bbox.height / fig.dpi)


def _wrap_lines(text: str, width_inches: float, fontsize: float) -> List[str]:
    """``text`` broken into lines that fit ``width_inches`` at ``fontsize``, measured with the
    font matplotlib draws it in (a character count over- or under-fills proportional fonts)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(size=fontsize)
    limit = max(width_inches, 1.0) * 72.0 * 0.97  # points, with a little slack for kerning

    def fits(line: str) -> bool:
        return TextPath((0, 0), line.replace("$", r"\$"), prop=prop).get_extents().width <= limit

    lines: List[str] = []
    for paragraph in str(text).split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}" if current else word
            if current and not fits(candidate):
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def _draw_badges(
    ax: Any, explanation: Any, items: Sequence[Mapping[str, Any]], height: int, width: int, box: _PlotBox
) -> None:
    """Number badges at each region's anchor on ``ax`` (an image of ``height × width`` pixels)."""
    points, kept = [], []
    for item in items:
        if item.get("anchor") is not None:
            x, y = item["anchor"]
            points.append((x * width - 0.5, y * height - 0.5))
        else:
            key = item["id"] if item.get("id") is not None else item["name"]
            try:
                mask = region_mask(explanation, key, height, width)
            except (KeyError, ValueError, TypeError):
                continue  # no region map for this name: leave the number off the picture
            if not mask.any():
                continue
            points.append(_label_point(mask))
        kept.append(item)
    spacing = _BADGE_INCHES * box.font * width / max(box.panel_inches, 1e-6) * 1.05
    for item, (x, y) in zip(kept, _spread(points, spacing, width, height)):
        colour = BAR_POSITIVE if item["group"] == "for" else BAR_NEGATIVE
        ax.text(x, y, str(item["number"]), ha="center", va="center", fontsize=_BADGE_SIZE * box.font,
                fontweight="bold", color="white", bbox=_badge_box(colour), zorder=5)


def _badge_box(colour: str) -> Dict[str, Any]:
    return {"boxstyle": "circle,pad=0.28", "facecolor": colour, "edgecolor": "white", "linewidth": 1.3}


def _spread(points: Sequence[Tuple[float, float]], spacing: float, width: int, height: int) -> List[Tuple[float, float]]:
    """Nudge badge centres apart until no two are closer than ``spacing`` pixels (a few rounds
    of pairwise pushes), keeping every badge on the picture."""
    placed = [[float(x), float(y)] for x, y in points]
    margin = spacing / 2
    for _ in range(40):
        moved = False
        for i in range(len(placed)):
            for j in range(i):
                dx, dy = placed[i][0] - placed[j][0], placed[i][1] - placed[j][1]
                distance = math.hypot(dx, dy)
                if distance >= spacing:
                    continue
                if distance < 1e-6:
                    dx, dy, distance = 0.0, 1.0, 1.0
                push = (spacing - distance) / 2 + 0.01
                placed[i][0] += dx / distance * push
                placed[i][1] += dy / distance * push
                placed[j][0] -= dx / distance * push
                placed[j][1] -= dy / distance * push
                moved = True
        for point in placed:
            point[0] = min(max(point[0], margin - 0.5), width - 0.5 - margin)
            point[1] = min(max(point[1], margin - 0.5), height - 0.5 - margin)
        if not moved:
            break
    return [(x, y) for x, y in placed]


def _draw_region_chart(fig: Any, box: _PlotBox, explanation: Any, items: Sequence[Mapping[str, Any]], subject: str) -> None:
    """The numbered regions as up to two labelled groups of bars, right of the photos."""
    method = explanation.get("method") or "gradcam"
    groups = [
        ("for", "Strongest regions" if method == "gradcam" else f"For {subject}", _FOR_INK),
        ("against", f"Against {subject}", _AGAINST_INK),
    ]
    rows: List[Tuple[float, Optional[Tuple[str, str]], Optional[Mapping[str, Any]]]] = []
    y = 0.0
    for key, header, ink in groups:
        members = [item for item in items if item["group"] == key]
        if not members:
            continue
        if rows:
            y -= 1.5  # air between the groups
        rows.append((y, (header, ink), None))
        for item in members:
            y -= 1.0
            rows.append((y, None, item))
    depth = -y + 1.2 if rows else 2.0
    s = box.font
    # Row pitch: comfortable, but the chart, its axis label and the note must fit beside the photos.
    pitch = min(0.34 * s, max(box.chart_span - 1.6 * s, 0.5) / depth)
    chart_height = pitch * depth
    bottom = box.top - chart_height
    labels_ax = fig.add_axes(box.rect(box.chart_left, bottom, box.labels, chart_height))
    bars_ax = fig.add_axes(box.rect(box.chart_left + box.labels, bottom, box.bars - 0.08, chart_height))
    # The label column draws over the bars' (blank) background, so a long header stays whole.
    labels_ax.set_zorder(bars_ax.get_zorder() + 1)
    for ax in (labels_ax, bars_ax):
        ax.set_ylim(y - 0.6, 0.6)
    labels_ax.set_xlim(0.0, box.labels)  # x in inches, so badge and name positions are fixed
    labels_ax.set_axis_off()
    heading = "Which regions matter\nnumbers match the badges on the heatmap" if items else "Regions"
    fig.text(box.chart_left / box.width, (box.top + 0.08 * s) / box.height, heading, ha="left", va="bottom",
             fontsize=_CAPTION_SIZE * s, linespacing=1.3)
    if not items:
        labels_ax.text(0.0, -0.5, "No regions in this record", ha="left", va="center", fontsize=_LABEL_SIZE * s,
                       color=_MUTED_INK)
        bars_ax.set_axis_off()
        return

    most_for = max([item["attribution"] for item in items if item["attribution"] > 0], default=0.0)
    most_against = max([-item["attribution"] for item in items if item["attribution"] < 0], default=0.0)
    # Each side that has bars keeps ~0.45 inch free for its value labels, so a label never
    # leaves the axes and never reaches the region names in the column to its left.
    room = min(0.45 / max(box.bars - 0.08, 0.5), 0.3)
    sides = int(most_for > 0) + int(most_against > 0)
    extent = (most_for + most_against) / max(1.0 - room * sides, 0.2) or 1.0
    right = most_for + room * extent if most_for > 0 else 0.03 * extent
    left = -(most_against + room * extent) if most_against > 0 else -0.03 * extent
    bars_ax.set_xlim(left, right)
    pad = 0.015 * (right - left)
    for y_row, group_header, row_item in rows:
        if group_header is not None:
            title, title_ink = group_header
            labels_ax.text(0.0, y_row, title, ha="left", va="center", fontsize=_CAPTION_SIZE * s, fontweight="bold",
                           color=title_ink)
            continue
        if row_item is None:
            continue
        item = row_item
        value = float(item["attribution"])
        support = item["group"] == "for"
        strongest = most_for if support else most_against
        shade = _mix(*(SUPPORT_GRADIENT if support else AGAINST_GRADIENT), abs(value) / strongest if strongest else 1.0)
        labels_ax.text(0.15 * s, y_row, str(item["number"]), ha="center", va="center", fontsize=_BADGE_SIZE * s,
                       fontweight="bold", color="white", bbox=_badge_box(BAR_POSITIVE if support else BAR_NEGATIVE))
        labels_ax.text(0.34 * s, y_row, item["name"], ha="left", va="center", fontsize=_LABEL_SIZE * s,
                       color=_MUTED_INK)
        bars_ax.barh([y_row], [value], height=0.62, color=shade)
        bars_ax.text(value + (pad if value >= 0 else -pad), y_row, _short(value), ha="left" if value >= 0 else "right",
                     va="center", fontsize=(_LABEL_SIZE - 1) * s, color=_VALUE_INK)
    # The zero line runs through each group's bars only, not through the group headers.
    for key in ("for", "against"):
        ys = [y_row for y_row, _, item in rows if item is not None and item["group"] == key]
        if ys:
            bars_ax.vlines(0.0, min(ys) - 0.5, max(ys) + 0.5, color="#8a8f98", linewidth=0.8)
    bars_ax.set_yticks([])
    bars_ax.tick_params(axis="x", labelsize=8.5 * s, colors=_MUTED_INK)
    bars_ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    bars_ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bars_ax.spines[side].set_visible(False)
    axis_label = _bar_axis_label(explanation, subject, most_against > 0)
    bars_ax.set_xlabel(axis_label, fontsize=9 * s, color=_VALUE_INK, linespacing=1.3)
    note = "\n".join(_wrap_lines(_chart_note(explanation, subject), box.labels + box.bars - 0.1, 9.0 * s))
    below = (0.62 + 0.17 * axis_label.count("\n")) * s  # tick labels and the axis label, in inches
    fig.text(box.chart_left / box.width, (bottom - below) / box.height, note, ha="left", va="top", fontsize=9.0 * s,
             color=_MUTED_INK, linespacing=1.35)



def _bar_axis_label(explanation: Any, subject: str, against: bool) -> str:
    if (explanation.get("method") or "gradcam") == "gradcam":
        return "mean Grad-CAM heat (0 to 1)"

    units = "change in the prediction" if _is_regression(explanation) else "change in log-odds"
    if against:
        return f"← against {subject}   |   for {subject} →\n{units}"
    return f"{units} for {subject}"



def _chart_note(explanation: Any, subject: str) -> str:
    """One or two plain sentences under the chart: what a bar measures, in the record's terms."""
    options = _options(explanation)
    method = explanation.get("method") or "gradcam"
    hidden = "blurred" if options.get("background") == "blur" else "grayed"
    if method == "gradcam":
        return (
            f"Grad-CAM heat is similar across the regions it marks, so read these as the regions the "
            f"model uses most for {subject}, not as a ranking."
        )
    scale = "in the model's own units" if _is_regression(explanation) else "in log-odds (+1 multiplies the odds by 2.7)"
    if method == "occlusion":
        return f"Each bar is how much the score for {subject} falls when that region and its neighbours are {hidden}, {scale}."
    if method == "lime":
        return (
            f"Each bar is what showing the region adds to the score for {subject}, fitted over many copies "
            f"of the photo with random regions {hidden}, {scale}."
        )
    return f"Each bar is the region's attribution for {subject}, {scale}."


def _mix(start: str, end: str, t: float) -> Tuple[float, float, float]:
    """The colour ``t`` of the way from ``start`` to ``end`` (hex strings)."""
    a, b = _hex_rgb(start), _hex_rgb(end)
    t = min(max(float(t), 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _label_point(mask: Any) -> Tuple[float, float]:
    """A point well inside ``mask`` (``x, y`` in pixels): erode until one more step would empty
    it, then take the remaining pixel nearest their mean. A centroid can fall outside a curved
    superpixel; this point never does. (Current records carry such a point as each region's
    ``anchor``; this is the fallback for older ones.)"""
    np = _numpy()
    core = mask
    while True:
        inner = core.copy()
        inner[1:, :] &= core[:-1, :]
        inner[:-1, :] &= core[1:, :]
        inner[:, 1:] &= core[:, :-1]
        inner[:, :-1] &= core[:, 1:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        if not inner.any():
            break
        core = inner
    ys, xs = np.nonzero(core)
    nearest = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    return float(xs[nearest]), float(ys[nearest])


def _hex_rgb(colour: str) -> Tuple[float, float, float]:
    value = colour.lstrip("#")
    return (int(value[0:2], 16) / 255.0, int(value[2:4], 16) / 255.0, int(value[4:6], 16) / 255.0)


# --- rendering internals -----------------------------------------------------------------------


def _options(explanation: Any) -> Mapping[str, Any]:
    if not isinstance(explanation, Mapping):
        raise TypeError("Expected an explanation record (a mapping with 'options').")
    options = explanation.get("options")
    return options if isinstance(options, Mapping) else {}


def _value_range(feature: Any) -> Sequence[float]:
    if feature is None:
        return (0.0, 1.0)
    spec = image_feature(feature)
    return tuple(float(v) for v in (spec.get("value_range") or (0.0, 1.0)))


def _upsample_kind(method: Any) -> str:
    # Grad-CAM is a continuous map and reads best smoothed; region methods must stay blocky so a
    # pixel always shows the value of the region it belongs to.
    return "bilinear" if (method or "gradcam") == "gradcam" else "nearest"


def _evidence_fill(options: Mapping[str, Any], base: Any) -> Any:
    """What the evidence check put in place of hidden pixels: a blurred copy or flat gray."""
    evidence = options.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("fill") != "blur":
        return _EVIDENCE_GRAY
    sigma = evidence.get("blur_sigma") or options.get("blur_sigma")
    if sigma is None:
        sigma = _BLUR_SIGMA_224 * min(base.shape[0], base.shape[1]) / 224.0
    return gaussian_blur(base, float(sigma))


def _base_pixels(image: Any, value_range: Sequence[float]) -> Any:
    """``h × w × 3`` floats in ``[0, 1]`` from a PIL image, a nested list or an array."""
    np = _numpy()
    if hasattr(image, "convert") and hasattr(image, "size"):
        base = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    else:
        low, high = (float(v) for v in value_range)
        base = (np.asarray(image, dtype=float) - low) / ((high - low) or 1.0)
        if base.ndim == 2:
            base = base[..., None]
        if base.ndim != 3:
            raise ValueError(f"image must be h × w × c, got shape {base.shape}.")
        base = np.repeat(base[..., :1], 3, axis=-1) if base.shape[-1] == 1 else base[..., :3]
    return np.clip(base, 0.0, 1.0)


def _nearest_index(m: int, n: int) -> Any:
    """Source index for each of ``n`` destination pixels: ``floor(dst · m / n)``."""
    np = _numpy()
    return np.minimum((np.arange(n) * m) // n, m - 1)


def _bilinear_index(m: int, n: int) -> Tuple[Any, Any, Any]:
    """Half-pixel-centre sampling: ``(lower index, upper index, fraction)`` for ``n`` pixels."""
    np = _numpy()
    src = np.clip((np.arange(n) + 0.5) * (m / n) - 0.5, 0.0, m - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, m - 1)
    return lo, hi, src - lo


def _ramp(a: Any, stops: Sequence[Sequence[float]]) -> Any:
    np = _numpy()
    s0, s1, s2 = (np.asarray(s, dtype=float) for s in stops)
    lower = s0 + (s1 - s0) * (a / 0.5)[..., None]
    upper = s1 + (s2 - s1) * ((a - 0.5) / 0.5)[..., None]
    return np.where((a <= 0.5)[..., None], lower, upper)


def _outline(mask: Any) -> Any:
    """Edge pixels of a boolean mask: any pixel whose 4-neighbour (inside the image) differs."""
    np = _numpy()
    edge = np.zeros_like(mask, dtype=bool)
    vertical = mask[1:, :] != mask[:-1, :]
    horizontal = mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    return edge


def _compose(base: Any, field: Any, mode: str, threshold: float, outline: bool) -> Any:
    """Spec B steps 2–3: paint ``field`` over ``base`` in ``mode`` and outline the focus."""
    np = _numpy()
    if mode == "heatmap":
        alpha = heat_alpha(field)[..., None]
        out = base * (1.0 - alpha) + heat_color(field) * alpha
    elif mode == "spotlight":
        lit = np.clip((field - (threshold - _SPOT_FADE)) / _SPOT_FADE, 0.0, 1.0)
        out = base * (_SPOT_DIM + (1.0 - _SPOT_DIM) * lit)[..., None]
    elif mode == "original":
        return base
    else:
        raise ValueError(f"mode must be 'heatmap', 'spotlight' or 'original', got {mode!r}.")
    if outline and threshold > 0:
        edge = _outline(field >= threshold)[..., None]
        out = np.where(edge, out * (1.0 - _OUTLINE_MIX) + _OUTLINE_MIX, out)
    return out


def _to_image(pixels: Any) -> Any:
    """Floats in ``[0, 1]`` to a PIL image with the shared rounding rule, ``floor(x·255 + 0.5)``."""
    np = _numpy()
    Image, _ = _pil()
    data = np.clip(np.floor(pixels * 255.0 + 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(data)


def _region_id(options: Mapping[str, Any], region: Union[str, int]) -> int:
    if isinstance(region, int) and not isinstance(region, bool):
        return region
    for item in options.get("regions") or []:
        if isinstance(item, Mapping) and str(item.get("name")) == str(region):
            return int(item["id"])
    if str(region).isdigit():
        return int(region)
    raise KeyError(f"No region named {region!r} in this explanation.")


def _grid_segments(options: Mapping[str, Any], region: Union[str, int]) -> Tuple[Any, int]:
    """Older records: rebuild the patch grid and find the patch ``region`` names."""
    np = _numpy()
    grid = options.get("grid") or 8
    rows, cols = (int(grid), int(grid)) if isinstance(grid, (int, float)) else (int(grid[0]), int(grid[1]))
    ids = np.arange(rows * cols).reshape(rows, cols)
    if isinstance(region, int) and not isinstance(region, bool):
        return ids, region
    for r in range(rows):
        for c in range(cols):
            name = f"{_PATCH_ROWS[r]} {_PATCH_COLS[c]}" if (rows, cols) == (4, 4) else f"row {r} col {c}"
            if name == str(region):
                return ids, r * cols + c
    raise KeyError(f"No region named {region!r} in this explanation.")


def _position_word(cx: float, cy: float) -> str:
    row = _POSITION_ROWS[min(int(cy * 3), 2)]
    col = _POSITION_COLS[min(int(cx * 3), 2)]
    if row == "middle":
        return "centre" if col == "centre" else f"{col} side"
    if col == "centre":
        return f"{row} middle"
    return f"{row} {col}"


def _is_regression(explanation: Any) -> bool:
    """Whether a record scores a raw value (regression) rather than class probabilities."""
    space = _options(explanation).get("score_space")
    if space in ("value", "log_odds"):
        return space == "value"
    output = explanation.get("output") or {}
    return not output.get("probabilities")


def _target_probability(explanation: Any) -> Optional[float]:
    output = explanation.get("output") or {}
    value = output.get("value")
    if value is None:
        probabilities = output.get("probabilities") or {}
        target = _options(explanation).get("target") or output.get("target_label") or output.get("label")
        value = probabilities.get(target) if isinstance(probabilities, Mapping) else None
        if value is None:
            value = output.get("probability")
    return None if value is None else float(value)


def _percent(p: float) -> str:
    if p >= 0.9995:
        return "99.9%+"
    if p < 0.0005:
        return "<0.1%"
    return f"{p * 100:.1f}%"


def _format_value(value: float) -> str:
    """A regression output with three significant figures (``12.3``, ``0.00457``, ``1.23e+04``)."""
    return f"{float(value):.3g}"


def _short(value: float) -> str:
    return f"{value:+.2f}" if abs(value) >= 0.1 else f"{value:+.3f}"


# --- loading internals -------------------------------------------------------------------------


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
