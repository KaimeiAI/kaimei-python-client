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

The same call as `POST https://api.kaimei.ai/v1/explanations`. See the [product site](https://kaimeiai-website.pages.dev/) for the language-agnostic REST examples.

## Install

```bash
pip install kaimei
```

To extract and upload a fitted estimator you also need NumPy (and your training stack):

```bash
pip install "kaimei[models]"
```

For a local API (the `kaimei-api` service in the [KaimeiAI/kaimei](https://github.com/KaimeiAI/kaimei) repo):

```python
client = Kaimei(api_key="km_test_...", base_url="http://localhost:8000")
```

`KAIMEI_API_KEY` and `KAIMEI_BASE_URL` are read from the environment when arguments are omitted.

Create an API key in the [dashboard](https://dashboard.kaimei.ai) (session login). Keys cannot create other keys.

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

A runnable walkthrough lives in [`examples/upload_sklearn.py`](examples/upload_sklearn.py) and [`examples/upload_sklearn.ipynb`](examples/upload_sklearn.ipynb).

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

To explain a new photo, `kaimei.images` (`pip install "kaimei[images]"`) reads it at the model's input size and paints the result:

```python
from kaimei.images import load_image, overlay_heatmap

model = client.models.retrieve("cats-dogs-photos")
photo = load_image("my_dog.jpg", model=model)  # a path, bytes, a PIL image or an array
explanation = client.explanations.create(model=model.slug, input={"photo": photo}, method="gradcam")
print(explanation.output["label"], explanation.options["layer"])
overlay_heatmap(photo, explanation.options["heatmap"]).save("my_dog_gradcam.png")
```

`load_image` scales the shorter side to the model's size and crops the middle, exactly like torchvision's `Resize(n)` + `CenterCrop(n)` (`crop="stretch"` resizes instead), applies the photo's EXIF rotation, and returns pixels in the feature's `value_range`. The dashboard's Explain tab prepares uploaded photos the same way. NumPy arrays can also be passed straight into any request's `input`. [`examples/cats_dogs_gradcam.py`](examples/cats_dogs_gradcam.py) trains a MobileNetV2 cat/dog classifier on the Oxford-IIIT Pet photos, uploads it, and runs Grad-CAM on new photos.

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
