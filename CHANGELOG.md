# Changelog

## 0.4.0 — 2026-10-24

Convolutional image models upload exactly, from PyTorch and Keras, and photos load and explain
from Python.

### Added

- PyTorch CNNs are traced with `torch.fx` (forward hooks remain the fallback), so residual adds,
  `torch.cat`, squeeze-excitation gates, `torch.flatten` and functional activations are kept.
  `Conv2d` with any stride / padding / dilation / groups, `BatchNorm2d`, max / average /
  adaptive pooling, `ZeroPad2d`, `ReLU6`, `LeakyReLU`, `ELU`, `Hardswish`, `Hardsigmoid`.
  ResNet, MobileNet v2 / v3, EfficientNet, DenseNet, VGG and SqueezeNet replay to within 1e-7 of
  PyTorch. Pass `input_shape=(channels, height, width)`.
- Keras functional graphs (and nested base models, as in transfer learning) keep their branches:
  `DepthwiseConv2D`, `SeparableConv2D`, `BatchNormalization`, average and global pooling, `Add`,
  `Multiply`, `Concatenate`, `ZeroPadding2D`, `Rescaling`, `Normalization`, strided and dilated
  `Conv2D` with `"same"` padding. MobileNetV2, ResNet50 and EfficientNetB0 replay exactly.
- `preprocess={"mean": [...], "std": [...]}` bakes input normalization into the graph;
  `value_range=[0, 255]` declares the pixel units an image model reads.
- Classifiers that end in logits get a softmax / sigmoid appended so Kaimei scores probabilities.
- `upload(X=...)` accepts image batches, `(n, h, w, c)` or `(n, c, h, w)`, and keeps the reference
  set under the API's value budget.
- `client.projects.list(include="summary")` / `retrieve(..., include="summary")` return each
  project's roll-up (models, 30-day traffic, alerts), and `client.projects.overview(project)` returns
  the project dashboard: per-model traffic and drift, open alerts, the feature × model attribution
  matrix, and recent analyses and explanations.
- `kaimei.images.load_image(source, size=None, *, model=None)` reads a photo (a path, bytes, a PIL
  image or an array) into the nested list an image feature expects: the shorter side scaled to fit
  and the middle cropped, exactly as torchvision's `Resize` + `CenterCrop`, with the EXIF rotation
  applied and pixels in the feature's `value_range`. `model=` takes the size from a model record.
- `kaimei.images.overlay_heatmap(image, heatmap)` paints an explanation's `options["heatmap"]` over
  the image, as the dashboard draws it.
- `pip install "kaimei[images]"` installs NumPy and Pillow for these helpers.
- `examples/cats_dogs_gradcam.py`: train a MobileNetV2 cat/dog classifier on the Oxford-IIIT Pet
  photos, upload it, and run Grad-CAM on new photos.

### Changed

- An operation that cannot be replayed and changes the tensor shape raises
  `StructureError(code="unsupported_layers")` naming it, instead of an approximate graph that
  could not compile.
- Request bodies may contain NumPy arrays and scalars; they are sent as plain JSON lists and numbers.
- With the default `externalize_above`, a spec that would still be 2 MB of JSON or more (a CNN's
  many mid-sized kernels: MobileNetV2 was 16 MB) also lifts tensors above 4,096 values to binary,
  so it uploads as 1.4 MB. A threshold you pass is used as given.

## 0.3.0 — 2026-10-23

Models with millions of parameters and thousands of layers upload and render.

### Added

- Tensors with more than `externalize_above` values (default 65,536) leave the JSON spec and
  upload as zlib-compressed float32 blobs (`PUT /structure/tensors/{key}`), referenced from the
  spec as `{"$tensor": key}`. `client.models.upload()` registers the model from its shapes,
  streams the tensors, then sends the real spec; `progress=` reports each stage.
- `weights="summary"` uploads shapes and statistics only. The dashboard draws the architecture
  while the weights stay in your environment.
- `extract_structure()` returns an `Extraction` (spec plus tensors); `estimator_to_spec()` is
  unchanged and keeps everything inline.
- PyTorch and Keras layers carry their module `path`, so the API groups them into blocks and
  detects repeated blocks (`×N`).
- `client.models.tensors()`, `delete_tensor()`, `structure_parameter_values()` (binary window),
  `structure(root=, depth=, detail=)` and `trace_structure(detail="summary", nodes=)`.

## 0.2.2 — 2026-10-13

### Fixed

- mypy type-checks as Python 3.10 and resolves feature names to `list[str]` before
  reading a wrapped estimator.

## 0.2.1 — 2026-10-13

### Added

- `estimator_to_spec` unwraps wrappers around a fitted sklearn / Keras / Torch estimator
  (`model_`, `estimator_`, `best_estimator_`, `module`, and the same family) without
  special-casing vendor packages. Framework labels follow the core-library MRO.

## 0.2.0 — 2026-10-12

First release that can upload a fitted estimator from a notebook.

### Added

- `client.models.upload()` extracts sklearn / Keras / PyTorch structure and registers it in one call.
- `estimator_to_spec`, reference-data helpers, and a supported-estimator matrix in the README.
- Structure-backed inference on the API: executable uploads can be explained, predicted, and analysed without a remote URL.
- `client.health()`, `Page.auto_paging_iter()`, and `*.iterate()` helpers.
- Retries on 429 (and GET 502/503/504), honouring `Retry-After`.
- `auth.signup(..., turnstile_token=)` plus `verify_email` / `resend_verification`.
- Examples: `examples/upload_sklearn.py` and `examples/upload_sklearn.ipynb`.
- `PermissionDeniedError` (keeps `PermissionError` as an alias).

### Changed

- Version bump from 0.1.0. Default HTTP timeout is 120 seconds.
- `upload()` no longer PUTs `/structure` after `POST /v1/models` already stored it.
- Reference rows are capped at 5,000 with a warning. Large JSON specs warn at 2 MB and 16 MB.
- Non-executable uploads (importance summaries, approximate nets) warn that explain/predict need a remote URL.

### Packaging

- Extra `kaimei[models]` installs NumPy.
- CI runs Ruff, mypy, and Python 3.9–3.13.
