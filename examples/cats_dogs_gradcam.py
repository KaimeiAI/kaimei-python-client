"""Train a cats-vs-dogs photo classifier, upload it to Kaimei, and explain new photos.

The model is torchvision's MobileNetV2, pretrained on ImageNet, with a new one-logit head
trained on the Oxford-IIIT Pet photos (7,349 photos of 37 breeds: 12 cat, 25 dog). Kaimei
replays every uploaded layer itself, so Grad-CAM, occlusion and LIME run in the API on any
photo you send, from Python or from the model's Explain tab on the dashboard.

    pip install "kaimei[images]" matplotlib torch torchvision
    set KAIMEI_API_KEY=km_live_...
    python examples/cats_dogs_gradcam.py train     # downloads the photos (~800 MB) on first run
    python examples/cats_dogs_gradcam.py upload    # registers the model with the example photos attached
    python examples/cats_dogs_gradcam.py examples  # (re)attach the example photos to a registered model
    python examples/cats_dogs_gradcam.py explain examples/photos/cat_siamese.jpg examples/photos/dog_pug.jpg
    python examples/cats_dogs_gradcam.py explain examples/photos/cat_and_dog.jpg --target cat --method lime

``train`` keeps the pretrained convolutional layers frozen and fits the head, which takes a
few minutes on a laptop CPU. ``upload`` registers the model and attaches ``examples/photos``
as clickable example photos of the image feature, so the dashboard's Explain tab works without
uploading anything. ``explain`` prints the summary sentence, the evidence check and the numbered
regions (for the class, then against it), and saves ``<photo>_<method>.png`` into ``--out``: the
dashboard's view, with the heatmap and its numbered badges, the spotlight, the evidence pictures
and the region chart. The notebook ``examples/cats_dogs_explain.ipynb`` walks through the same
pictures.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from kaimei import Kaimei, NotFoundError
from kaimei.images import (
    explanation_summary,
    image_feature,
    load_image,
    numbered_regions,
    plot_explanation,
)

SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LABELS = ["cat", "dog"]  # Oxford-IIIT Pet's binary category: 0 = cat, 1 = dog
WEIGHTS_FILE = "cats_dogs_mobilenet_v2.pt"
#: Example photos shipped next to this script (Oxford-IIIT Pet test split, see photos/README.md).
PHOTOS = Path(__file__).resolve().parent / "photos"
#: Dashboard example size: 224 px JPEGs keep the model record small (~10 KB per photo).
EXAMPLE_SIZE = 224
EXAMPLE_QUALITY = 80


def build_model(pretrained: bool = True):
    import torchvision
    from torch import nn

    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.mobilenet_v2(weights=weights)
    # One logit: P(dog) = sigmoid(logit). Kaimei appends the sigmoid when the model is uploaded.
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.last_channel, 1))
    return net


def pets(root: str, split: str, transform=None):
    from torchvision.datasets import OxfordIIITPet

    return OxfordIIITPet(root, split=split, target_types="binary-category", transform=transform, download=True)


def train(args: argparse.Namespace) -> None:
    import torch
    from torch import nn
    from torchvision import transforms

    normalize = transforms.Normalize(MEAN, STD)
    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    # The same resize + center crop that kaimei.images.load_image and the dashboard apply.
    evaluate = transforms.Compose([transforms.Resize(SIZE), transforms.CenterCrop(SIZE), transforms.ToTensor(), normalize])
    train_set = pets(args.data, "trainval", augment)
    test_set = pets(args.data, "test", evaluate)
    targets = np.asarray(train_set._bin_labels)
    n_cats, n_dogs = int((targets == 0).sum()), int((targets == 1).sum())
    print(f"train: {n_cats} cats, {n_dogs} dogs; test: {len(test_set)} photos")

    torch.manual_seed(0)
    net = build_model(pretrained=True)
    for param in net.features.parameters():
        param.requires_grad_(False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    optimizer = torch.optim.AdamW(net.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    # Dogs outnumber cats about 2 to 1; weight the classes equally.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_cats / n_dogs))
    for epoch in range(args.epochs):
        net.train()
        net.features.eval()  # frozen BatchNorm statistics
        started, total, seen = time.perf_counter(), 0.0, 0
        for images, labels in loader:
            logits = net(images).squeeze(1)
            loss = loss_fn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            seen += len(labels)
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / seen:.4f} ({time.perf_counter() - started:.0f}s)")

    net.eval()
    correct = np.zeros(2)
    count = np.zeros(2)
    with torch.no_grad():
        for images, labels in torch.utils.data.DataLoader(test_set, batch_size=128, num_workers=args.workers):
            predicted = (net(images).squeeze(1) > 0).long()
            for cls in (0, 1):
                mask = labels == cls
                count[cls] += int(mask.sum())
                correct[cls] += int((predicted[mask] == cls).sum())
    metrics = {
        "dataset": "Oxford-IIIT Pet (binary)",
        "backbone": "mobilenet_v2 (ImageNet), frozen",
        "train_photos": len(train_set),
        "test_photos": len(test_set),
        "test_accuracy": round(float(correct.sum() / count.sum()), 4),
        "test_accuracy_cat": round(float(correct[0] / count[0]), 4),
        "test_accuracy_dog": round(float(correct[1] / count[1]), 4),
        "epochs": args.epochs,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / WEIGHTS_FILE)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"saved {out / WEIGHTS_FILE}")


def client_from_env() -> Kaimei:
    if not os.environ.get("KAIMEI_API_KEY"):
        raise SystemExit("Set KAIMEI_API_KEY (and optionally KAIMEI_BASE_URL) before running.")
    return Kaimei(
        api_key=os.environ["KAIMEI_API_KEY"],
        base_url=os.environ.get("KAIMEI_BASE_URL") or "https://api.kaimei.ai",
        timeout=600,
    )


def example_photos(folder: Path = PHOTOS) -> list:
    """The shipped photos as image-feature examples: ``{"label", "image": "data:image/jpeg;base64,..."}``.

    The dashboard shows them as clickable thumbnails on the Explain tab. Each is center-cropped to
    the model's square and re-encoded small, so the model record stays a few dozen kilobytes.
    """
    from PIL import Image

    examples = []
    for path in sorted(folder.glob("*.jpg")):
        # "cat_british_shorthair" → "Cat · British Shorthair"; "cat_and_dog" → "Cat and dog".
        species, _, breed = path.stem.partition("_")
        label = f"{species.title()} · {breed.replace('_', ' ').title()}" if breed != "and_dog" else "Cat and dog"
        pixels = np.asarray(load_image(path, EXAMPLE_SIZE))
        buffer = io.BytesIO()
        Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(buffer, format="JPEG", quality=EXAMPLE_QUALITY)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        examples.append({"label": label, "image": f"data:image/jpeg;base64,{encoded}"})
    return examples


def with_examples(features: list, examples: list) -> list:
    """The feature list with ``examples`` attached to its image feature (other features untouched)."""
    updated = []
    for feature in features:
        feature = dict(feature)
        if feature.get("type") == "image":
            feature["examples"] = examples
        updated.append(feature)
    return updated


def upload(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out)
    net = build_model(pretrained=False)
    net.load_state_dict(torch.load(out / WEIGHTS_FILE, map_location="cpu"))
    net.eval()
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    client = client_from_env()
    if args.replace:
        try:
            client.models.delete(args.slug)
            print(f"deleted the existing {args.slug}")
        except NotFoundError:
            pass
    X = y = None
    if args.reference:
        # A few held-out photos as reference data, so the Insights tab can run an analysis.
        test_set = pets(args.data, "test")
        picks = np.random.default_rng(0).choice(len(test_set), size=args.reference, replace=False)
        X = np.asarray([load_image(test_set[int(i)][0], SIZE) for i in picks])
        y = [LABELS[test_set[int(i)][1]] for i in picks]

    # The image feature, with the example photos attached so the dashboard has something to click.
    features = [{"name": "photo", "type": "image", "height": SIZE, "width": SIZE, "channels": 3}]
    if PHOTOS.is_dir():
        features = with_examples(features, example_photos())
    model = client.models.upload(
        net,
        name="Cats vs dogs (photos)",
        slug=args.slug,
        feature_names=["photo"],
        features=features,
        task="binary_classification",
        outputs={"labels": LABELS, "positive_label": "dog", "threshold": 0.5},
        input_shape=(3, SIZE, SIZE),
        # torchvision models read Normalize(mean, std) on [0, 1] pixels; bake it into the graph.
        preprocess={"mean": MEAN, "std": STD},
        description="MobileNetV2 (ImageNet) with a cat/dog head trained on the Oxford-IIIT Pet photos.",
        tags=["vision", "pytorch", "grad-cam"],
        metadata=metrics,
        X=X,
        y=y,
        progress=lambda stage, done, total: print(f"  {stage} {done}/{total}", end="\r"),
    )
    print(f"\nregistered {model.slug} ({model.id})")
    print("on the dashboard, open the model's Explain tab, click an example photo and pick Grad-CAM")


def examples(args: argparse.Namespace) -> None:
    """Attach (or refresh) the example photos on an already-registered model, weights untouched."""
    client = client_from_env()
    model = client.models.retrieve(args.model)
    image_feature(model)  # fails early when the model has no image feature
    photos = example_photos(Path(args.photos))
    # to_dict() gives plain dicts (nested values included); dict(APIObject) would keep wrappers.
    current = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in model.features]
    updated = client.models.update(args.model, features=with_examples(current, photos))
    print(f"attached {len(photos)} example photos to {updated.slug}: " + ", ".join(p["label"] for p in photos))


def explain(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")  # files only; no window
    import matplotlib.pyplot as plt

    client = client_from_env()
    model = client.models.retrieve(args.model)
    feature = image_feature(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # A fixed seed makes LIME's random samples, and so its picture, the same on every run (the
    # notebook uses seed 0 too); Grad-CAM and occlusion are deterministic anyway.
    options = {"seed": 0, **({"target_class": args.target} if args.target else {})}
    for path in args.photos:
        photo = load_image(path, model=model)
        explanation = client.explanations.create(
            model=args.model, input={feature["name"]: photo}, method=args.method, options=options
        )
        output = explanation.output
        p_dog = float(output["probabilities"]["dog"])
        explained = explanation.options.get("target") or output.get("target_label")
        print(
            f"{path}: {output['label']} (P(dog) = {p_dog:.3f}), {args.method} for {explained}"
            f" on {explanation.options.get('layer') or explanation.options.get('segmentation', 'patches')},"
            f" {explanation.n_evals} model calls, {explanation.latency_ms or 0:.0f} ms"
        )
        print(f"  {explanation_summary(explanation)}")
        evidence = explanation.options.get("evidence")
        if evidence:
            hidden = "blurred" if evidence.get("fill") == "blur" else "hidden"
            print(
                f"  evidence: P({evidence['label']}) = {evidence['full']:.1%} for the photo, "
                f"{evidence['without_focus']:.1%} with the focus {hidden}, {evidence['focus_only']:.1%} for the focus alone"
            )
        print_regions(explanation, explained)
        stem = f"{Path(path).stem}_{args.method}{'_' + args.target if args.target else ''}"
        figure = plot_explanation(photo, explanation, feature=feature)
        figure.savefig(out / f"{stem}.png", dpi=110)
        plt.close(figure)
        print(f"  saved {out / f'{stem}.png'}")


def print_regions(explanation, target: str) -> None:
    """The numbered regions as the figure shows them: the strongest for the class, then against it."""
    areas = {region["name"]: region.get("area") for region in explanation.options.get("regions") or []}
    units = "mean heat" if explanation.method == "gradcam" else "log-odds"
    headers = {"for": "strongest regions" if explanation.method == "gradcam" else f"for {target}", "against": f"against {target}"}
    group = None
    for item in numbered_regions(explanation):
        if item["group"] != group:
            group = item["group"]
            print(f"  {headers[group]} ({units}):")
        area = areas.get(item["name"])
        share = f"  {area * 100:.0f}% of the photo" if area is not None else ""
        print(f"    {item['number']}  {item['name']:<15} {item['attribution']:+.3f}{share}")



def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="./data", help="where the Oxford-IIIT Pet photos are (or will be downloaded)")
    common.add_argument("--out", default="./cats-dogs-model", help="weights, metrics and heatmaps go here")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", parents=[common], help="fit the cat/dog head on the Oxford-IIIT Pet photos")
    fit.add_argument("--epochs", type=int, default=4)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--workers", type=int, default=4)
    send = sub.add_parser("upload", parents=[common], help="register the trained model with Kaimei")
    send.add_argument("--slug", default="cats-dogs-photos")
    send.add_argument("--reference", type=int, default=0, help="held-out photos to upload as reference data")
    send.add_argument("--replace", action="store_true", help="delete an existing model with this slug first")
    attach = sub.add_parser("examples", help="attach the example photos to a registered model (no re-upload)")
    attach.add_argument("--model", default="cats-dogs-photos")
    attach.add_argument("--photos", default=str(PHOTOS), help="folder of JPEGs named <species>_<breed>.jpg")
    ask = sub.add_parser("explain", parents=[common], help="explain new photos with the uploaded model")
    ask.add_argument("photos", nargs="+")
    ask.add_argument("--model", default="cats-dogs-photos")
    ask.add_argument("--method", default="gradcam", choices=["gradcam", "occlusion", "lime"])
    ask.add_argument("--target", choices=LABELS, help="explain this class instead of the predicted one")
    args = parser.parse_args()
    {"train": train, "upload": upload, "examples": examples, "explain": explain}[args.command](args)


if __name__ == "__main__":
    main()
