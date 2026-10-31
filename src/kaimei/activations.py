"""What every layer computed for one input: feature maps, firing units and class attention.

``client.models.activations(model, input={...})`` runs one input through an uploaded network
and returns, per stage, how strongly it fired and (for the detailed stages) the maps of its
strongest channels, a saliency map and optionally Grad-CAM at that layer. Maps travel as
base64 ``uint8`` bytes with their value range; these helpers turn them back into arrays and
draw them::

    from kaimei.activations import feature_maps, plot_activations, saliency, stage

    result = client.models.activations("cats-dogs-photos", input={"photo": photo}, channels=8)
    relu = stage(result, "features.18.2")       # one stage by node id (or the last feature map)
    maps = feature_maps(relu)                    # {channel: (h, w) array}, strongest first
    where = saliency(relu)                       # (h, w) mean |activation| over channels
    plot_activations(result, image=photo)        # the dashboard's Sight strip (matplotlib)

Class attention: ``client.models.activations(..., nodes=[layer], cam=[layer], target="dog")``
and :func:`class_attention` give the Grad-CAM heatmap at that layer, or ``None`` with the
API's reason in ``stage["cam"]["reason"]`` when it did not fit the time budget.

Needs NumPy; ``plot_activations`` also needs matplotlib. Importing this module loads neither.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "class_attention",
    "decode",
    "detailed_stages",
    "feature_maps",
    "plot_activations",
    "saliency",
    "spatial_stages",
    "stage",
    "unit_values",
]


def _np():
    import numpy

    return numpy


def _payload(result: Any) -> Mapping[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, Mapping):
        return result
    data = getattr(result, "_data", None)
    if isinstance(data, Mapping):
        return data
    return dict(result)


def decode(packed: Mapping[str, Any]):
    """A byte-packed array back to floats: ``value = lo + byte / 255 · (hi − lo)``, in ``shape``."""
    np = _np()
    raw = np.frombuffer(base64.b64decode(packed["data"]), dtype=np.uint8).astype(float)
    lo, hi = (float(bound) for bound in packed["range"])
    return (lo + raw / 255.0 * (hi - lo)).reshape([int(dim) for dim in packed["shape"]])


def _stages(result: Any) -> List[Mapping[str, Any]]:
    return [dict(item) for item in _payload(result).get("stages", [])]


def spatial_stages(result: Any) -> List[Dict[str, Any]]:
    """Stages that are maps (``height × width × channels``), input to output."""
    return [dict(item) for item in _stages(result) if len(item.get("shape") or []) == 3 and item.get("op") != "input"]


def detailed_stages(result: Any) -> List[Dict[str, Any]]:
    """Stages whose maps or values travelled with the response."""
    return [dict(item) for item in _stages(result) if item.get("detailed")]


def stage(result: Any, node_id: Optional[str] = None) -> Dict[str, Any]:
    """One stage by node id; without an id, the last detailed feature map (else the last stage)."""
    stages = _stages(result)
    if node_id is not None:
        for item in stages:
            if item.get("node_id") == node_id:
                return dict(item)
        names = ", ".join(str(item.get("node_id")) for item in stages[-8:])
        raise KeyError(f"No stage '{node_id}'. The last stages are: {names}.")
    maps = [item for item in stages if item.get("detailed") and len(item.get("shape") or []) == 3 and item.get("op") != "input"]
    if maps:
        return dict(maps[-1])
    if not stages:
        raise ValueError("The result has no stages.")
    return dict(stages[-1])


def feature_maps(item: Mapping[str, Any]) -> Dict[int, Any]:
    """``{channel: (h, w) array}`` of a detailed spatial stage, strongest channel first."""
    return {int(entry["channel"]): decode(entry) for entry in item.get("maps") or []}


def saliency(item: Mapping[str, Any]):
    """Mean ``|activation|`` over channels of a detailed spatial stage, or ``None``."""
    packed = item.get("saliency")
    return decode(packed) if packed else None


def unit_values(item: Mapping[str, Any]):
    """Every value of a detailed vector stage (up to 4,096 units), or ``None`` when too wide."""
    packed = item.get("values")
    return decode(packed) if packed else None


def class_attention(item: Mapping[str, Any]):
    """The Grad-CAM heatmap at this layer as an ``(h, w)`` array in ``[0, 1]``, or ``None``."""
    cam = item.get("cam")
    if not cam or not cam.get("available") or "data" not in cam:
        return None
    return decode(cam)


def _upsample(values, height: int, width: int):
    """Nearest-neighbour stretch of a map to the picture, for drawing only."""
    np = _np()
    grid = np.asarray(values, dtype=float)
    rows = np.minimum((np.arange(height) * grid.shape[0]) // height, grid.shape[0] - 1)
    cols = np.minimum((np.arange(width) * grid.shape[1]) // width, grid.shape[1] - 1)
    return grid[rows][:, cols]


def plot_activations(
    result: Any,
    image: Any = None,
    *,
    stages: Optional[Sequence[str]] = None,
    channels: int = 4,
    figure: Any = None,
):
    """Draw the strip the dashboard's Sight view shows: the picture, then one column per detailed
    feature map with its saliency over the picture and its strongest channel maps below.

    ``stages`` picks node ids (default: every detailed feature map, at most eight, evenly spaced);
    ``channels`` is how many maps to draw per stage. Returns the matplotlib figure.
    """
    np = _np()
    import matplotlib.pyplot as plt  # noqa: WPS433 (optional dependency)

    payload = _payload(result)
    picked = [stage(result, node_id) for node_id in stages] if stages else detailed_stages(result)
    picked = [item for item in picked if len(item.get("shape") or []) == 3 and item.get("op") != "input"]
    if not stages and len(picked) > 8:
        index = np.unique(np.linspace(0, len(picked) - 1, 8).round().astype(int))
        picked = [picked[i] for i in index]
    rows = 1 + max(0, min(channels, max((len(item.get("maps") or []) for item in picked), default=0)))
    cols = len(picked) + (1 if image is not None else 0)
    fig = figure if figure is not None else plt.figure(figsize=(2.1 * max(cols, 1), 2.1 * rows))
    axes = fig.subplots(rows, max(cols, 1), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()

    picture = None
    if image is not None:
        picture = np.asarray(image, dtype=float)
        if picture.ndim == 3 and picture.shape[2] == 1:
            picture = picture[:, :, 0]
        feature_range = payload.get("input") or {}
        peak = float(np.nanmax(picture)) if picture.size else 1.0
        if peak > 1.0:
            picture = picture / 255.0
        axes[0][0].imshow(np.clip(picture, 0, 1), cmap="gray" if picture.ndim == 2 else None)
        axes[0][0].set_title("input", fontsize=9)
        del feature_range
    offset = 1 if image is not None else 0
    for column, item in enumerate(picked):
        axis = axes[0][column + offset]
        where = saliency(item)
        label = f"{item.get('label', item.get('node_id'))}\n{'×'.join(str(d) for d in item['shape'])}"
        if where is not None:
            if picture is not None:
                axis.imshow(np.clip(picture, 0, 1), cmap="gray" if picture.ndim == 2 else None)
                heat = _upsample(where, picture.shape[0], picture.shape[1])
                axis.imshow(heat / (heat.max() or 1.0), cmap="magma", alpha=0.6, vmin=0, vmax=1)
            else:
                axis.imshow(where, cmap="magma")
        axis.set_title(label, fontsize=8)
        for row, (channel, grid) in enumerate(list(feature_maps(item).items())[: rows - 1], start=1):
            sub = axes[row][column + offset]
            signed = bool(grid.min() < 0 < grid.max())
            peak = float(np.abs(grid).max()) or 1.0
            sub.imshow(grid, cmap="RdBu_r" if signed else "magma", vmin=-peak if signed else 0, vmax=peak)
            sub.set_title(f"ch {channel}", fontsize=8)
    output = payload.get("output") or {}
    title = output.get("label")
    if title is not None and output.get("probability") is not None:
        fig.suptitle(f"{title} · P = {float(output['probability']):.3f}", fontsize=10)
    return fig
