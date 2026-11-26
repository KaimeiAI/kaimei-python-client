"""Upload a small segmentation network, then compare Kaimei's mask with the model's own.

A tiny PyTorch convolutional network learns to mark a bright disc on a noisy 32 x 32 picture. The pictures are
made below, so nothing is downloaded and training takes a few seconds on a laptop CPU.

    pip install kaimei torch numpy onnx
    export KAIMEI_API_KEY=km_live_...          # optional: export KAIMEI_BASE_URL=http://localhost:8000
    python examples/image_segmentation.py
"""

from __future__ import annotations

import base64
import os

import numpy as np
import torch
from torch import nn

from kaimei import Kaimei

# Step 1. Synthetic pictures: a bright disc of random size and place on a noisy background, and its mask.
SIZE = 32
rng = np.random.default_rng(0)
rows, cols = np.mgrid[0:SIZE, 0:SIZE]


def make_picture() -> tuple[np.ndarray, np.ndarray]:
    cy, cx = rng.uniform(8, SIZE - 8, size=2)
    radius = rng.uniform(4, 8)
    mask = ((rows - cy) ** 2 + (cols - cx) ** 2 <= radius**2).astype(np.float32)
    picture = 0.35 * rng.random((SIZE, SIZE)) + 0.45 * mask  # noise, plus a brighter disc
    return picture.astype(np.float32), mask


pairs = [make_picture() for _ in range(600)]
pictures = torch.tensor(np.stack([p for p, _ in pairs]))[:, None]  # (600, 1, 32, 32)
masks = torch.tensor(np.stack([m for _, m in pairs]))[:, None]


# Step 2. The model: three convolutions, one output channel, a sigmoid. A pixel is "disc" at 0.5 or more.
class Segmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


torch.manual_seed(0)
model = Segmenter()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(150):
    batch = torch.randint(0, len(pictures), (32,))
    loss = nn.functional.binary_cross_entropy(model(pictures[batch]), masks[batch])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()
print(f"The network was trained for {step + 1} steps; its last loss was {loss.item():.3f}.")

# Step 3. Connect to Kaimei.
client = Kaimei(
    api_key=os.environ["KAIMEI_API_KEY"],
    base_url=os.environ.get("KAIMEI_BASE_URL") or "https://api.kaimei.ai",
)

# Step 4. Upload it as a segmentation. sample= is an example picture, as you would call the model with it.
uploaded = client.models.upload(
    model,
    name="Disc segmenter",
    task="segmentation",
    sample=torch.rand(1, 1, SIZE, SIZE),
)
print(f"The model is registered as {uploaded.id}.")

# Step 5. Ask Kaimei how it read the model: exactly, and whether it runs the model itself.
structure = client.models.structure(uploaded.id, detail="summary")
print(
    f"Kaimei's reading is {structure['fidelity']} (check: {structure['verification']}); "
    f"it {'runs' if structure['runnable'] else 'only draws'} the model."
)

# Step 6. One prediction, on a new picture (height x width x channels, channels last).
picture, truth = make_picture()
prediction = client.models.predict(uploaded.id, input={"image": picture[:, :, None].tolist()})

# The mask comes back as one class number per pixel, packed as base64 bytes: 1 is the disc.
packed = prediction.output["mask"]
height, width = packed["shape"]
kaimei_mask = np.frombuffer(base64.b64decode(packed["data"]), dtype=np.uint8).reshape(height, width) == 1

# Step 7. The model's own mask is its sigmoid at 0.5 or more. Compare, then print Kaimei's mask as a text grid.
with torch.no_grad():
    own_mask = (model(torch.tensor(picture)[None, None])[0, 0] >= 0.5).numpy()
agree = int((kaimei_mask == own_mask).sum())
print(f"Kaimei's mask agrees with the model's own on {agree} of {own_mask.size} pixels.")
print(f"The model marks {int(own_mask.sum())} pixels as disc; the true disc has {int(truth.sum())}.")
print("Kaimei's mask (# disc, . background), every second row:")
for r in range(0, SIZE, 2):
    print("".join("#" if kaimei_mask[r, c] else "." for c in range(SIZE)))
