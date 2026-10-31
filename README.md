# kaimei

[![PyPI](https://img.shields.io/pypi/v/kaimei.svg)](https://pypi.org/project/kaimei/)
[![Python versions](https://img.shields.io/pypi/pyversions/kaimei.svg)](https://pypi.org/project/kaimei/)
[![License](https://img.shields.io/pypi/l/kaimei.svg)](https://github.com/KaimeiAI/kaimei-python-client/blob/main/LICENSE)

Official Python client for the [Kaimei](https://kaimei.ai) explainability API.

Kaimei (解明) — *to clarify, elucidate, reveal through investigation* — turns trained models into something you can inspect: architecture, feature-level attributions, global behaviour, learned rules, robustness, and drift.

```python
from kaimei import Kaimei

client = Kaimei(api_key="km_live_...")

explanation = client.explanations.create(
    model="credit-risk-v3",
    prediction_id="pred_8f2a19",
    input={
        "income": 72000,
        "credit_score": 681,
        "utilization": 0.41,
    },
)

print(explanation.output)
print(explanation.top_features)
# 0.81 decline  |  credit_score -0.22
```

The same call as `POST https://api.kaimei.ai/v1/explanations`. See [kaimei.ai](https://kaimei.ai) for the language-agnostic REST examples and the [API reference](https://api.kaimei.ai/docs) for every endpoint.

## Install

```bash
pip install kaimei
```

To extract and upload a fitted estimator you also need NumPy (and your training stack):

```bash
pip install "kaimei[models]"
```

For a self-hosted or local Kaimei API, pass its address:

```python
client = Kaimei(api_key="km_test_...", base_url="http://localhost:8000")
```

`KAIMEI_API_KEY` and `KAIMEI_BASE_URL` are read from the environment when arguments are omitted.

Create an API key in the [dashboard](https://dashboard.kaimei.ai) (session login). Keys cannot create other keys.

## Examples

- **[titanic-kaimei-example](https://github.com/KaimeiAI/titanic-kaimei-example)**: a complete workflow on the Titanic data, in five notebooks with their outputs (and Colab badges):
  - upload a scikit-learn model;
  - explain passengers, with counterfactuals and two models compared;
  - analyse the whole model;
  - monitor production traffic and catch drift;
  - ship an improved version;
  - explain a model Kaimei can't run itself: XGBoost, via TreeSHAP values and via a scoring endpoint.
- [`examples/upload_sklearn.py`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/upload_sklearn.py) / [`.ipynb`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/upload_sklearn.ipynb): the smallest upload-and-explain round trip.
- [`examples/cats_dogs_gradcam.py`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/cats_dogs_gradcam.py) and [`cats_dogs_explain.ipynb`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/cats_dogs_explain.ipynb): an image model (MobileNetV2) explained with Grad-CAM, occlusion and LIME.

## Upload a trained model from a notebook

The client reads the live estimator in your process and uploads layers and learned weights. Small tensors travel inline in the JSON spec; anything larger than 65,536 values is streamed as a zlib-compressed float32 blob and referenced from the spec, so models with millions of parameters upload at about 4 bytes per weight. Kaimei does **not** store the original `.h5` / `.pt` / pickle file, and it does not host a binary artifact.

```python
from kaimei import Kaimei

client = Kaimei(api_key="km_live_...")

model = client.models.upload(
    clf,  # a fitted estimator
    name="Churn net",
    feature_names=["tenure", "spend", "tickets"],  # optional if the estimator knows them
    X=X_train,   # optional reference rows for explanations (capped at 5,000)
    y=y_train,
)
# Open the dashboard — the architecture scene shows every layer and tensor.
```

What happens next:

- **Visualize** — always. The dashboard draws the uploaded structure.
- **Explain / predict / analyse** — when the spec is executable (linear, MLP, decision tree, or a forest of ≤30 trees / ≤4,000 nodes). The API runs that structure in-process.
- **Otherwise** — pass a live scoring URL, or POST pre-computed attributions:

```python
client.models.upload(
    large_forest,
    name="Churn forest",
    inference={"type": "remote", "url": "https://scoring.example.com/predict"},
)
```

Attach weights to a model you already registered with `client.models.upload_structure(model.id, clf)`.

### Large models

Nothing changes in the call. A PyTorch module with thousands of layers and millions of parameters uploads the same way; the client registers the model from its shapes, streams each large tensor (`progress=` reports `("tensors", done, total)`), then sends the spec that references them. Module names become block paths, so the dashboard folds `encoder.layers.0 … encoder.layers.47` into one `×48` block and loads a tensor only when you open it.

```python
model = client.models.upload(
    transformer,                 # torch.nn.Module, 1,200 layers / 110M parameters
    name="Encoder",
    feature_names=["token_id"],
    progress=lambda stage, done, total: print(stage, done, "/", total),
)
```

Options:

- `externalize_above=` — values per tensor above which it uploads as binary (default 65,536; `None` keeps everything inline).
- `weights="summary"` — upload shapes and statistics only. The architecture is drawable; the weights never leave your environment. Explain / predict then need `inference={"type": "remote", ...}`.
- `extract_structure(estimator, ...)` returns the spec and the tensors without uploading, for inspection or a custom upload.
- `client.models.tensors(model)` lists what was uploaded; `structure_parameter_values(model, "fc1.weight")` reads a tensor back as float32.

A runnable walkthrough lives in [`examples/upload_sklearn.py`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/upload_sklearn.py) and [`examples/upload_sklearn.ipynb`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/upload_sklearn.ipynb).

### Supported estimators

| Estimator | What is uploaded | Executable for explain / predict |
| --- | --- | --- |
| sklearn linear (`coef_` / `intercept_`) | Exact weights | Yes |
| sklearn `MLPClassifier` / `MLPRegressor` | Exact layers | Yes |
| sklearn tree | Full node list | Yes |
| sklearn forest / extra trees | Trees if ≤30 estimators and ≤4,000 nodes; otherwise importances | Yes only when trees are inlined |
| sklearn `Pipeline` + `StandardScaler` or `(0, 1)` `MinMaxScaler` | Scaler folded or inserted, then the estimator | Same as the final step |
| Keras / TensorFlow `Dense`, `Conv2D` (any stride, padding, dilation, groups), `DepthwiseConv2D`, `SeparableConv2D`, `BatchNormalization`, max / average / global pooling, `Add`, `Multiply`, `Concatenate`, `ZeroPadding2D`, `Rescaling`, `Normalization`, flatten, dropout, LayerNorm; functional graphs and nested base models | Layer weights and the layer graph | Yes, unless a layer is unrecognized (`approximate`) |
| `torch.nn` CNNs: `Conv2d` (stride, padding, dilation, groups), `BatchNorm2d`, max / average / adaptive pooling, residual `+`, `torch.cat`, squeeze-excitation `*`, `Flatten`, `ReLU6` / `Hardswish` / `SiLU` and the other common activations (ResNet, MobileNet, EfficientNet, DenseNet, VGG, SqueezeNet) | Traced with `torch.fx`, so functional ops are kept | Yes; pass `input_shape=(C, H, W)` |
| `torch.nn` `Linear`, common activations, Dropout, LayerNorm, Embedding, BatchNorm1d | Forward-hook order | Yes, unless a layer is unrecognized (`approximate`) |
| Wrapper around any of the above (`model_`, `estimator_`, `best_estimator_`, `module`, …) | Same as the inner estimator | Same as the inner estimator |
| XGBoost / LightGBM / CatBoost (sklearn API) | `feature_importances_` summary | No — pass `inference` remote or attributions |
| ONNX, Hugging Face, LLMs | Framework label only | No |

Unknown Keras / PyTorch layers that keep their input's shape are passed through and marked `approximate`. Those models visualize but cannot be queried unless you add a remote URL. Operations that change the shape and cannot be replayed (for example ShuffleNet's channel shuffle or an upsampling decoder) raise `StructureError(code="unsupported_layers")` naming them; register such models with a remote inference URL instead.

### Results in a notebook

A record that ends a Jupyter cell (an explanation, an insight, a model, a prediction, an analysis, or a page of them) shows as a Kaimei card: the mark, the dashboard's colours and the dashboard's numbers, formatted by the same rules. An explanation card has the prediction, the attribution bars (hover a bar for its full precision), the probabilities, the counterfactual, and the input and JSON behind collapsible sections; an image explanation adds the photo, the evidence heatmap and the focus, with its regions. Insight cards draw importance shares, drift by status, behaviour curves, the output histogram and the learned rules the way the dashboard's cards do. The cards are plain HTML and CSS, so they render in JupyterLab, classic Jupyter, VS Code and Colab, follow the host's dark theme, and link back to the record in the dashboard.

```python
explanation = client.explanations.create(model="titanic-forest", input=passenger, method="shap")
explanation            # the card
```

For the plain data instead:

```python
explanation.to_dict()  # the JSON the API returned, as a dict
explanation.to_json()  # the same, as a string
explanation.to_frame() # the dashboard's table as a pandas DataFrame
#    rank feature value  attribution   share direction
# 0     1     Sex  male       -0.091  0.5112  negative
# 1     2  Pclass     3       -0.042  0.2360  negative
# ...
```

`to_frame()` gives the record's main table: one row per feature for an explanation (the API's `top_features` rows as sent, then the remaining `attributions` with `share` and `rank` by the same formula; an image explanation lists its regions), the payload's table for an insight (importance shares, PSI per feature, interaction pairs, fragile features, rules, one row per grid point of each behaviour curve, histogram bins), the features of a model, and one flattened row per item for a page (`output.label`, `attributions.Age`, ...). `kaimei.display.rows(record)` is the same without pandas.

If you never want the cards:

```python
import kaimei
kaimei.set_display("raw")     # cells show the JSON (a collapsible tree in JupyterLab); "rich" restores the cards

with kaimei.display.using("raw"):
    explanation               # raw for this block only
```

`KAIMEI_DISPLAY=raw` in the environment sets the default, `kaimei.display.html(record)` returns a card's HTML for a report of your own, and `KAIMEI_DASHBOARD_URL` points the cards' links at a self-hosted dashboard.

### Tabular explanations in a notebook

`kaimei.tabular` turns explanations and insights into sentences and charts in the dashboard's colours (`pip install matplotlib` for the charts):

```python
from kaimei.tabular import (
    compare_explanations, counterfactual_summary, explanation_summary,
    plot_counterfactual, plot_explanation, plot_insight,
)

model = client.models.retrieve("titanic-forest")
explanation = client.explanations.create(
    model=model.slug, input=passenger, method="shap",
    options={"counterfactual": True, "frozen_features": ["Sex", "Age"]},
)
print(explanation_summary(explanation, model=model))
# The model predicts died: P(survived) = 7.4%, against 41.1% on average over the reference data.
# Pulling P(survived) down: Sex = male (−12.1 pts), Title = Mr (−11.0 pts), Pclass = 3 (−4.5 pts).
plot_explanation(explanation, model=model)     # waterfall from the average to this prediction
print(counterfactual_summary(explanation, model=model))
plot_counterfactual(explanation, model=model)  # the smallest change that flips the decision, step by step
compare_explanations([explanation, other_model_explanation], ["Forest", "Logistic"], model=model)
for insight in client.insights.list(model=model.slug).data:
    plot_insight(insight, model=model)         # importance, rules, behaviour curves, interactions, drift, ...
```

The examples above come from [titanic-kaimei-example](https://github.com/KaimeiAI/titanic-kaimei-example). Passing `model=` adds feature units (`Fare = 7.25 GBP`) and names the output (`P(survived)`).

For a model Kaimei runs itself with up to 12 inputs, KernelSHAP enumerates every feature coalition over a fixed reference sample. The answer is therefore exact for that sample, and the same request returns the same numbers.

### Image models

Images travel as nested `height × width × channels` lists, channels last, with pixels in `[0, 1]` unless the feature says otherwise. A PyTorch CNN needs its input size (PyTorch order); a Keras model reports its own.

```python
import torchvision

net = torchvision.models.resnet18(num_classes=10).eval()
client.models.upload(
    net,
    name="Defect classifier",
    task="multiclass_classification",
    outputs={"labels": ["ok", "scratch", "dent", ...]},
    input_shape=(3, 224, 224),
    # torchvision-style models expect Normalize on [0, 1] pixels; bake it into the graph:
    preprocess={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    X=images,  # optional reference images, (n, 3, 224, 224) or (n, 224, 224, 3)
)

# A Keras model with a Rescaling(1/255) layer reads 0–255 pixels:
client.models.upload(keras_model, name="Leaf disease", value_range=[0, 255])

client.explanations.create(model="defect-classifier", input={"image": pixels}, method="gradcam")
client.explanations.create(model="defect-classifier", input={"image": pixels}, method="occlusion",
                           options={"grid": 8, "target_class": "dent"})
```

`gradcam` reads the last convolutional feature map of an uploaded structure (`options={"layer": ...}` picks another). `occlusion` and `lime` work for any image model, including a remote endpoint; that endpoint receives images as nested lists by default, or as `{"b64": ...}` PNG / float32 with `inference={"image_encoding": "png"}`, channels first with `"image_layout": "chw"`, and normalized with `"image_normalize": {"mean": [...], "std": [...]}`. Requests are split to stay under `max_request_bytes` (8 MB).

To explain a new photo, `kaimei.images` (`pip install "kaimei[images]"`) reads it at the model's input size and draws the result exactly as the dashboard's Explain tab does:

```python
from kaimei.images import load_image, render_explanation, explanation_summary, plot_explanation

model = client.models.retrieve("cats-dogs-photos")
photo = load_image("my_dog.jpg", model=model)  # a path, bytes, a PIL image or an array
explanation = client.explanations.create(model=model.slug, input={"photo": photo}, method="gradcam")
print(explanation_summary(explanation))
# "The model looks mostly at the centre of the photo: half of its evidence for dog sits in 21% of the
#  picture. That region alone is enough to say dog (P(dog) = 97.5%); with it hidden, P(dog) falls
#  from 99.2% to 48.1%."
render_explanation(photo, explanation).save("my_dog_gradcam.png")                   # warm glow + focus outline
render_explanation(photo, explanation, mode="spotlight").save("my_dog_spotlight.png")  # the photo, dimmed outside the focus
plot_explanation(photo, explanation)  # matplotlib: the dashboard's view, with numbered regions (below)
```

`load_image` scales the shorter side to the model's size and crops the middle, exactly like torchvision's `Resize(n)` + `CenterCrop(n)` (`crop="stretch"` resizes instead), applies the photo's EXIF rotation, and returns pixels in the feature's `value_range`. The dashboard's Explain tab prepares uploaded photos the same way. NumPy arrays can also be passed straight into any request's `input`.

Heatmaps use the dashboard's palette: amber → orange → red where the photo is evidence *for* the explained class, light blue → indigo where it is evidence *against* it (the SHAP convention), never more than 66% opaque so the eyes and nose under the hot spot stay visible. A white outline marks the focus region, the smallest part of the photo holding half of the evidence. Grad-CAM maps are smoothed; occlusion and LIME keep one flat value per region.

Image explanations carry, besides `options["heatmap"]`, the regions the attributions refer to (`options["regions"]`: superpixels named by position, `upper left`, `centre 3`, …, each with its attribution, area, centroid and an `anchor` point for a number badge; `options["segments"]` maps heatmap cells to regions), the focus region (`options["focus"]`), an evidence check (`options["evidence"]`: P(target) for the full photo, with the focus hidden and with only the focus shown) and a plain-English `options["summary"]`. Photo-sized models hide regions by blurring them (`options["background"] == "blur"`, `evidence["fill"] == "blur"`, with `blur_sigma`), because flat gray is not neutral to a real model; thumbnails under 64 pixels keep gray.

The helpers render all of it with NumPy and Pillow only, byte for byte like the dashboard: `render_explanation(image, explanation, mode=...)`; `overlay_heatmap(image, heatmap, method=..., mode=..., threshold=...)` for raw heatmaps (without `method` it infers one: a map with negative values is a region map, anything else Grad-CAM); `evidence_images(image, explanation)` for the three evidence pictures, blurred with `gaussian_blur` (the API's exact blur) or grayed, as the API did; `numbered_regions(explanation)` for the regions to call out, numbered 1..N: up to five *for* the class, strongest first, then up to three *against* it; `focus_mask` / `region_mask(explanation, "upper left", h, w)` for pixel masks; and the building blocks `upsample_heatmap`, `heat_color`, `heat_alpha` and `focus_threshold`. `plot_explanation` (needs matplotlib) draws the dashboard's view: the photo, the heatmap with those numbers as badges at the regions' anchors, and the spotlight; below, the evidence pictures with their probabilities; on the right, a bar chart of the same numbered regions ("For cat" in orange, "Against cat" in blue; Grad-CAM lists its strongest regions), with the summary as the title. It returns the figure; pass `figure=` (a figure or subfigure) to draw several in one. Regression records print values with three significant figures instead of probabilities. `import kaimei.images` loads neither Pillow nor matplotlib.

[`examples/cats_dogs_gradcam.py`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/cats_dogs_gradcam.py) trains a MobileNetV2 cat/dog classifier on the Oxford-IIIT Pet photos, uploads it with the photos in [`examples/photos`](https://github.com/KaimeiAI/kaimei-python-client/tree/main/examples/photos) attached as clickable examples, and explains new photos (`explain` prints the summary, the evidence check and the numbered regions, and saves the figure). [`examples/cats_dogs_explain.ipynb`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/cats_dogs_explain.ipynb) asks the model *why* on those photos and draws every answer as the dashboard does: Grad-CAM on a cat and a dog, "why cat?" and "why dog?" on the cat-and-dog composite (each question lights up its own animal and marks the other one against), Grad-CAM, occlusion and LIME side by side on the beagle, and a gallery in which Grad-CAM finds the face on every photo while LIME singles out the ears and eyes of the cats and the noses of the dogs.

#### What the layers see

`client.models.activations` runs one input through an uploaded network and returns what every layer computed, the data behind the dashboard's **Sight** view: per stage how strongly it fired and which channels or units led; for the detailed stages (the API picks up to 24 feature maps plus the output, or pass `nodes=[...]`) each channel's energy, the maps of the strongest channels, a saliency map showing where in the picture the layer responds and, for narrow vectors, every value. `cam=[layer]` adds class attention (Grad-CAM) at that layer toward `target=` when it fits the API's time budget; early layers of a big network can be too slow, and the stage then says why.

```python
from kaimei.activations import class_attention, feature_maps, plot_activations, saliency, stage

result = client.models.activations("cats-dogs-photos", input={"photo": photo}, channels=8)
print(result.output["label"], [s["node_id"] for s in result.stages if s["detailed"]])
last = stage(result)                 # the last detailed feature map
maps = feature_maps(last)            # {channel: (h, w) array}, strongest first
where = saliency(last)               # (h, w): mean |activation| over channels
plot_activations(result, image=photo)  # matplotlib: the Sight strip

focused = client.models.activations("cats-dogs-photos", input={"photo": photo},
                                    nodes=[last["node_id"]], cam=[last["node_id"]], target="dog")
heat = class_attention(stage(focused, last["node_id"]))  # (H, W) Grad-CAM in [0, 1], or None
```

The JSON spec warns above ~2 MB; lower `externalize_above` if you see it. Each binary tensor may hold up to 250M values (the API's default cap on a single upload is 512 MB compressed).

## What you can do

```python
# Register a model (or point Kaimei at a remote inference URL / built-in demo)
client.models.create(
    name="Credit risk v3",
    slug="credit-risk-v3",
    task="binary_classification",
    features=[
        {"name": "income", "type": "numeric"},
        {"name": "credit_score", "type": "integer"},
        {"name": "utilization", "type": "numeric"},
    ],
    inference={"type": "builtin", "name": "credit-risk"},
)

# Log a prediction your own stack already made
client.predictions.create(model="credit-risk-v3", input={...}, output={"score": 0.81})

# Explain one input. method is "shap" (KernelSHAP), "shapley", or "lime".
client.explanations.create(model="credit-risk-v3", input={...}, method="shap")

# Study the model as a whole, then read the insights
analysis = client.analyses.create(model="credit-risk-v3", type="full")
client.analyses.wait(analysis.id)
insights = client.insights.list(model="credit-risk-v3")

# Walk every page
for model in client.models.iterate():
    print(model.slug)
```

Dashboard sign-in (session token) is also supported via `client.auth.login(...)` / `client.auth.signup(...)`. Production sign-up needs a Cloudflare Turnstile token (`turnstile_token=...`). Prefer creating API keys in the dashboard.

## Resources

| Attribute | Endpoints |
| --- | --- |
| `client.models` | `/v1/models` |
| `client.predictions` | `/v1/predictions` |
| `client.explanations` | `/v1/explanations` |
| `client.analyses` | `/v1/analyses` |
| `client.insights` | `/v1/insights` |
| `client.auth` | `/v1/auth` |
| `client.api_keys` | `/v1/api-keys` |
| `client.organization` | `/v1/organization` |
| `client.projects` | `/v1/projects` (`include="summary"`, `overview(project)`) |
| `client.health()` | `/health` |

Errors raise `kaimei.KaimeiError` subclasses (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, …) and expose `.code`, `.type`, and `.status_code`. `PermissionError` remains an alias of `PermissionDeniedError`.

429 responses are retried (honouring `Retry-After`). GET requests also retry 502/503/504. Default timeout is 120 seconds.

## License

Apache 2.0. Copyright Kaimei AI.
