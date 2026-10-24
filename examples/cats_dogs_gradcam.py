"""Train a cats-vs-dogs photo classifier, upload it to Kaimei, and run Grad-CAM on new photos.

The model is torchvision's MobileNetV2, pretrained on ImageNet, with a new one-logit head
trained on the Oxford-IIIT Pet photos (7,349 photos of 37 breeds: 12 cat, 25 dog). Kaimei
replays every uploaded layer itself, so Grad-CAM, occlusion and LIME run in the API on any
photo you send, from Python or from the model's Explain tab on the dashboard.

    pip install "kaimei[images]" torch torchvision
    set KAIMEI_API_KEY=km_live_...
    python examples/cats_dogs_gradcam.py train     # downloads the photos (~800 MB) on first run
    python examples/cats_dogs_gradcam.py upload
    python examples/cats_dogs_gradcam.py explain my_cat.jpg my_dog.jpg
    python examples/cats_dogs_gradcam.py explain both.jpg --target cat

``train`` keeps the pretrained convolutional layers frozen and fits the head, which takes a
few minutes on a laptop CPU. ``explain`` saves ``<photo>_gradcam.png`` (the heatmap painted
over the photo, as the dashboard shows it) into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

from kaimei import Kaimei, NotFoundError
from kaimei.images import image_feature, load_image, overlay_heatmap

SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LABELS = ["cat", "dog"]  # Oxford-IIIT Pet's binary category: 0 = cat, 1 = dog
WEIGHTS_FILE = "cats_dogs_mobilenet_v2.pt"


def build_model(pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.mobilenet_v2(weights=weights)
    # One logit: P(dog) = sigmoid(logit). Kaimei appends the sigmoid when the model is uploaded.
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.last_channel, 1))
    return net


def pets(root: str, split: str, transform=None):
    from torchvision.datasets import OxfordIIITPet

    return OxfordIIITPet(root, split=split, target_types="binary-category", transform=transform, download=True)


def train(args: argparse.Namespace) -> None:
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


def upload(args: argparse.Namespace) -> None:
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

    model = client.models.upload(
        net,
        name="Cats vs dogs (photos)",
        slug=args.slug,
        feature_names=["photo"],
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
    print("on the dashboard, open the model's Explain tab, upload a photo and pick Grad-CAM")


def explain(args: argparse.Namespace) -> None:
    client = client_from_env()
    model = client.models.retrieve(args.model)
    feature = image_feature(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    options = {"target_class": args.target} if args.target else None
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
            f" on {explanation.options.get('layer', 'patches')}, {explanation.latency_ms or 0:.0f} ms"
        )
        target = out / f"{Path(path).stem}_{args.method}{'_' + args.target if args.target else ''}.png"
        overlay_heatmap(photo, explanation.options["heatmap"], value_range=feature.get("value_range") or (0, 1)).save(target)
        print(f"  saved {target}")


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
    ask = sub.add_parser("explain", parents=[common], help="explain new photos with the uploaded model")
    ask.add_argument("photos", nargs="+")
    ask.add_argument("--model", default="cats-dogs-photos")
    ask.add_argument("--method", default="gradcam", choices=["gradcam", "occlusion", "lime"])
    ask.add_argument("--target", choices=LABELS, help="explain this class instead of the predicted one")
    args = parser.parse_args()
    {"train": train, "upload": upload, "explain": explain}[args.command](args)


if __name__ == "__main__":
    main()
