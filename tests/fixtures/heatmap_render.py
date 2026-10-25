"""Generate ``heatmap_render.json``: small rendering cases shared with the dashboard.

Both the Python helpers (``kaimei.images``) and the dashboard's TypeScript renderer implement the
same specification (upsampling, colour ramp, spotlight, focus outline, byte rounding, evidence
thumbnails). This file pins the expected bytes for a few tiny inputs so either side can prove it
still agrees::

    python tests/fixtures/heatmap_render.py        # rewrite the JSON from the current Python code

``tests/test_image_helpers.py`` re-renders every case and fails when the JSON drifts, so regenerate
it only after an intentional change to the rendering rules (and update the dashboard too).

Every byte is exact except the evidence thumbnails of the ``fill: "blur"`` cases: the blur's
weights come from ``exp``, and JavaScript's ``Math.exp`` may differ from NumPy's in the last
unit of precision, so a JavaScript renderer may be off by one there (``tolerance`` in the case).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "heatmap_render.json"

#: A 6 × 6 base image: a left-to-right ramp in red, top-to-bottom in green, mid blue, with four
#: decimals so JSON carries the exact floats both renderers start from.
BASE: List[List[List[float]]] = [
    [[round(x / 5, 4), round(y / 5, 4), round(0.25 + 0.5 * ((x + y) % 2), 4)] for x in range(6)] for y in range(6)
]

GRADCAM = [[1.0, 0.2], [0.0, 0.6]]
SIGNED = [[0.9, -0.5, 0.1], [0.3, 1.0, -1.0], [0.0, -0.2, 0.55]]
#: Every alpha / colour breakpoint of the palette (|v| = 0, 0.12, 0.42, 0.5, 0.72, 1), both signs.
RAMP = [[0.0, 0.12, 0.42, 0.5, 0.72, 1.0], [-0.0, -0.12, -0.42, -0.5, -0.72, -1.0]]

#: ``threshold: None`` means "use the fallback for the method" (0.5 for Grad-CAM, half of the
#: largest positive value otherwise), exactly as a record from before focus existed is drawn.
#: ``evidence`` (optional) is the record's ``options.evidence``: ``fill: "blur"`` hides with
#: ``gaussian_blur(base, blur_sigma)`` instead of gray 0.5 in the evidence thumbnails.
CASES: List[Dict[str, Any]] = [
    {"name": "gradcam_heatmap_default_threshold", "heat": GRADCAM, "method": "gradcam", "mode": "heatmap", "threshold": None},
    {"name": "gradcam_spotlight", "heat": GRADCAM, "method": "gradcam", "mode": "spotlight", "threshold": 0.5},
    {"name": "gradcam_low_threshold_outline", "heat": GRADCAM, "method": "gradcam", "mode": "heatmap", "threshold": 0.25},
    {"name": "occlusion_heatmap_default_threshold", "heat": SIGNED, "method": "occlusion", "mode": "heatmap", "threshold": None},
    {"name": "lime_spotlight", "heat": SIGNED, "method": "lime", "mode": "spotlight", "threshold": 0.3},
    {"name": "occlusion_original", "heat": SIGNED, "method": "occlusion", "mode": "original", "threshold": None},
    {"name": "lime_palette_breakpoints", "heat": RAMP, "method": "lime", "mode": "heatmap", "threshold": 0.45},
    {
        "name": "occlusion_blur_evidence",
        "heat": SIGNED,
        "method": "occlusion",
        "mode": "heatmap",
        "threshold": 0.3,
        "evidence": {"fill": "blur", "blur_sigma": 1.2},
    },
    {
        "name": "gradcam_blur_evidence_radius_capped",
        "heat": GRADCAM,
        "method": "gradcam",
        "mode": "spotlight",
        "threshold": 0.4,
        "evidence": {"fill": "blur", "blur_sigma": 3.0},
    },
]


def render_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """The case plus its expected bytes (``expected``) and evidence thumbnails."""
    import numpy as np

    from kaimei.images import evidence_images, overlay_heatmap

    image = overlay_heatmap(
        BASE, case["heat"], method=case["method"], mode=case["mode"], threshold=case["threshold"]
    )
    record: Dict[str, Any] = {"method": case["method"], "options": {"heatmap": case["heat"]}}
    if case["threshold"] is not None:
        record["options"]["focus"] = {"threshold": case["threshold"]}
    if case.get("evidence"):
        record["options"]["evidence"] = case["evidence"]
    _, without, only = evidence_images(BASE, record)
    out = dict(case)
    out["base"] = BASE
    out["expected"] = np.asarray(image).astype(int).ravel().tolist()
    out["without_focus"] = np.asarray(without).astype(int).ravel().tolist()
    out["focus_only"] = np.asarray(only).astype(int).ravel().tolist()
    if (case.get("evidence") or {}).get("fill") == "blur":
        # JavaScript's Math.exp may differ from NumPy in the last ulp: allow ±1 in the thumbnails.
        out["tolerance"] = {"expected": 0, "without_focus": 1, "focus_only": 1}
    return out


def build() -> Dict[str, Any]:
    return {
        "format": "kaimei.heatmap_render.v1",
        "note": (
            "RGB bytes are row-major (y, x, channel) for a 6x6 image; see kaimei/images.py and the rendering spec. "
            "Bytes are exact, except the evidence thumbnails of cases with a 'tolerance' (blurred hiding), which a "
            "JavaScript renderer may miss by 1 because Math.exp can differ from NumPy's exp in the last ulp."
        ),
        "value_range": [0.0, 1.0],
        "size": [6, 6],
        "cases": [render_case(case) for case in CASES],
    }


def main() -> None:
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    FIXTURE.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
