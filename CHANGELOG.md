# Changelog

## 0.7.1 — Unreleased

### Fixed

- Notebook cards keep their layout on GitHub and in other viewers that drop `<style>` blocks from
  HTML outputs. Each element now repeats its resting style inline, reading the theme's colours with
  the light theme as the fallback, and the logo and charts carry size attributes. Before, GitHub
  showed the logo as wide as the page and the rest unstyled. The stylesheet still drives the dark
  theme, hover and open sections in Jupyter, VS Code and Colab.
- GitHub's viewer also removes `<details>` and `<summary>`, which unfolded every section. The
  sections still read well open, except the JSON dumps: those are now hidden where the stylesheet
  is missing and shown, folded, everywhere else.
- The what-if steps of an explanation card name the probability they report: `→ P(survived) 33.9%
  · died` rather than `→ 33.9% died`, which read as the probability of dying. The before/after line
  says `P(survived) 7.4% → 33.9%` instead of `Output`.
- List cards (pages of models, explanations, predictions or insights) are wider, so every column
  fits; before, the last columns of a model list scrolled out of view.
- mypy passes again on the card renderer (CI).

## 0.7.0 — 2026-10-31

### Added

- `client.models.activations(model, input=...)`: what every layer computed for one input
  (`POST /v1/models/{id}/structure/activations`). Every stage reports its intensity, active share and
  strongest channels or units; the detailed stages carry each channel's energy, the maps of the
  strongest channels, a saliency map and, for narrow vectors, every value. `nodes=` picks the stages
  to detail, `channels=` how many maps, `cam=[layer]` adds class attention (Grad-CAM) at a layer
  toward `target=`, within the API's time budget.
- `kaimei.activations`: `decode` for the byte-packed maps, `stage`, `feature_maps`, `saliency`,
  `unit_values`, `class_attention`, and `plot_activations` (matplotlib), which draws the dashboard's
  Sight strip: the picture, then each detailed feature map's saliency over it and its strongest
  channel maps.
- A record that ends a Jupyter cell (an explanation, an insight, a model, a prediction, an
  analysis, a page of any of them, or anything else with an `object` type) displays as a Kaimei
  card: the mark, the dashboard's colours and type, and the dashboard's numbers formatted by the
  same rules (`fmtSigned`, `pct`, `outputParts`, `compact`, ... ported name for name).
  - An explanation shows the prediction (`died · P(died) = 92.6%`), the method, base value, model
    calls and latency, a one-sentence summary, the diverging attribution bars with the input value
    beside each feature (hover for six decimals; the rest of the features fold away), the class
    probabilities, the counterfactual step by step, and the input and the JSON behind collapsible
    sections. An image explanation shows the photo, the evidence heatmap and the focus (NumPy and
    Pillow, when installed), the evidence check and its numbered regions.
  - An insight shows its severity and score, the summary, and the payload as the dashboard draws
    it: importance and interaction shares, drift by status with the PSI, behaviour curves, the
    output histogram around the threshold (hover a bin), robustness, the learned rules.
  - A model shows its task, framework, status, counts, outputs and features; a prediction its
    output and input; an analysis its status, progress and timings; a page a table of its items
    with the total and the window shown.
  - The cards are HTML and CSS only (no script), render in JupyterLab, classic Jupyter, VS Code and
    Colab, follow the host's dark theme, and link to the record in the dashboard
    (`KAIMEI_DASHBOARD_URL` for a self-hosted one). `kaimei.display.html(record)` returns the HTML.
- The plain data: `record.to_json()`; `record.to_frame()` (pandas) and `kaimei.display.rows(record)`
  (no pandas) give the record's main table: one row per feature of an explanation (the API's
  `top_features` rows as sent, then the remaining attributions ranked with `share` by the same
  formula; regions for an image explanation), an insight payload's table (importance, drift,
  interactions, fragile features, rules, one row per grid point of each behaviour curve, histogram
  bins), a model's features, one flattened row per item of a page (`output.label`,
  `attributions.Age`, ...).
- `kaimei.set_display("raw")` makes cells show the JSON instead (a collapsible tree in JupyterLab);
  `kaimei.display.using("raw")` does it for one block; `KAIMEI_DISPLAY=raw` sets the default.
  Nested values such as `explanation.output` keep their plain repr in every mode.

## 0.6.3 — 2026-10-25

### Fixed

- `kaimei.tabular` chart titles use bold, which every matplotlib install has. Newer matplotlib
  logged `findfont: Failed to find font weight semibold for DejaVu Sans` under each chart.

### Documentation

- README: an Examples section (the [Titanic example](https://github.com/KaimeiAI/titanic-kaimei-example), the scikit-learn and image examples); links to [kaimei.ai](https://kaimei.ai) and the [API reference](https://api.kaimei.ai/docs) instead of a repository that isn't public.
- Package metadata: the "API" link points at the public API reference, plus an "Examples" link.

## 0.6.2 — 2026-10-25

### Fixed

- `kaimei.tabular.plot_insight` on a drift insight: the "moderate" and "significant" labels of the
  PSI reference lines overlapped when one input had drifted far (the lines sit close together on a
  wide axis). Each label now sits on the outer side of its line.
- `plot_explanation` for regression-style outputs (for example TreeSHAP values in log-odds) labels
  every bar with one shared precision, three significant figures of the largest, so a tiny
  attribution reads `+0.00` rather than `+0.000554`.

## 0.6.1 — 2026-10-25

### Fixed

- A request that timed out after reaching the server was retried whatever its method. A slow
  `POST` could therefore run twice or more: a batch of predictions that outlived the timeout was
  logged four times. Requests are now retried after a network error only when they are safe to
  repeat (`GET`, `HEAD`, `OPTIONS`) or never reached the server (connection refused or timed out).
  Otherwise `APIConnectionError` says the request was not retried because the server may already
  have applied it. Retries after 429, and `GET` retries after 502/503/504, are unchanged.

## 0.6.0 — 2026-10-25

Tabular explanations and model insights, drawn and worded the way the dashboard shows them.

### Added

- `kaimei.tabular`, for models whose inputs are table rows:
  - `explanation_summary(explanation, model=None)` states the prediction against the model's average
    and names the inputs that moved it most, in percentage points of `P(target)` for classifiers and
    in output units for regressions.
  - `plot_explanation(explanation, model=None, max_features=10)` draws a waterfall from the average
    prediction (the base value) to this one. Bars push toward the explained class (blue) or away
    (pink); the smallest attributions fold into one row; a hairline marks a binary model's threshold.
  - `counterfactual_summary` and `plot_counterfactual` show the counterfactual search step by step:
    the changes it made, the prediction after each, and whether the decision flipped. Requires an
    explanation created with `options={"counterfactual": True}`.
  - `compare_explanations([e1, e2], ["Model A", "Model B"])` puts one input's attributions from up
    to three models side by side.
  - `plot_insight(insight)` / `plot_insights(page)` draw every insight type an analysis produces:
    - importance, interactions and robustness as ranked bars;
    - behaviour curves as small multiples with individual-row curves;
    - the prediction histogram around the threshold;
    - drift, with PSI coloured by status and labelled in words;
    - the learned rules.
  - Passing `model=` (from `client.models.retrieve`) adds feature units (`Age = 22 years`) and names
    the output (`P(survived)`).
  - The text helpers need only the client. The plots need matplotlib, which `import kaimei.tabular`
    does not load.

### Changed

- A nested object without an `object` type, such as `explanation.output` or an item of
  `explanation.top_features`, now prints as its data instead of `<Kaimei object ...>`.

## 0.5.0 — 2026-10-25

Image explanations are drawn the way the dashboard draws them, and read like a sentence.

The summary, the numbered regions, the focus region, the evidence check and blurred hiding read
fields that the matching Kaimei API release adds to image explanations (`options["summary"]`,
`regions[].anchor`, `focus`, `evidence`, `background`). Against an older API the helpers still
work from the heatmap alone: the summary is built from it, regions follow the patch grid, and the
evidence pictures are marked "not measured".

### Added

- `kaimei.images.render_explanation(image, explanation, mode="heatmap" | "spotlight" | "original")`
  renders an explanation record from its `method`, `heatmap` and `focus`, in the dashboard's
  palette: amber → orange → red where the photo is evidence *for* the explained class, light
  blue → indigo where it is evidence *against* it (the SHAP convention), at most 66% opaque so the
  features under the hot spot stay visible; a white outline around the focus region (the smallest
  region holding half of the evidence); or the photo dimmed outside that region. Grad-CAM maps are
  smoothed with half-pixel-centre bilinear upsampling; occlusion / LIME region maps stay blocky.
- `explanation_summary(explanation)` returns the API's plain-English `options["summary"]` (and builds
  one from the heatmap for older records; regression records speak of "the prediction" with three
  significant figures). `evidence_images(image, explanation)` returns the photo, the photo with its
  focus hidden and the photo with only the focus kept — the pictures behind `options["evidence"]`,
  hidden the way the API hid them: blurred when `evidence["fill"] == "blur"`, grayed otherwise.
- `gaussian_blur(image, sigma)`: the API's blur for hiding regions (radius `ceil(3σ)`, mirrored
  edges, separable, float64), NumPy only and identical to the server's to 1e-12.
- `numbered_regions(explanation, for_count=5, against_count=3)`: the regions to call out, numbered
  1..N across two groups — the strongest *for* the class, then the strongest *against* it (Grad-CAM:
  its strongest regions only) — each with its name, attribution and `anchor` (a point inside the
  region for a number badge).
- `focus_mask`, `region_mask(explanation, name_or_id, h, w)` and `focus_threshold` expose the masks;
  `upsample_heatmap`, `heat_color` and `heat_alpha` are the building blocks. All of it is NumPy +
  Pillow and byte-identical to the dashboard (`tests/fixtures/heatmap_render.json` pins the shared
  cases; blurred evidence thumbnails may differ from a JavaScript renderer by one byte), and
  `import kaimei.images` loads neither Pillow nor matplotlib.
- `plot_explanation(image, explanation, *, modes=("heatmap", "spotlight"), figure=None, title=None,
  for_regions=5, against_regions=3)` draws the dashboard's view with matplotlib (optional): the
  photo with the model's answer, the heatmap with numbered badges on the strongest regions, and the
  spotlight; below, the evidence check (full photo, focus blurred or hidden, only the focus) with
  the model's probability for each; on the right, spanning both rows, a bar chart of the same
  numbered regions ("For cat" in orange, "Against cat" in blue) with a plain sentence saying what a
  bar measures. The summary is the title, wrapped to the actual figure or subfigure width.
  `BAR_POSITIVE` / `BAR_NEGATIVE` (`#f97316` / `#5b5bf5`) and `SUPPORT_GRADIENT` /
  `AGAINST_GRADIENT` are the matching chart colours.
- `examples/photos/`: six Oxford-IIIT Pet test photos (three cats, three dogs) and a cat-and-dog
  composite, with credits. `examples/cats_dogs_gradcam.py upload` attaches them to the image feature
  as clickable dashboard examples, the new `examples` subcommand attaches them to a model that is
  already registered, and `explain` prints the summary, the evidence check and the numbered regions
  and saves the full figure.
- `examples/cats_dogs_explain.ipynb`: Grad-CAM on a cat and a dog, "why cat?" and "why dog?" with
  LIME on the cat-and-dog composite, Grad-CAM, occlusion and LIME side by side on the beagle (and
  why perturbation methods can disagree with Grad-CAM), and a Grad-CAM + LIME gallery of all six
  photos, with the narrative written from the executed outputs. Figures are inlined as JPEG so the
  notebook stays small.

### Changed

- `overlay_heatmap(image, heatmap)` keeps its signature but draws the new look (graded glow, focus
  outline) and gains `method`, `mode`, `threshold` and `outline`; `opacity` is accepted and ignored.
  `method=None` (the default) infers the upsampling from the values: a map with any negative value
  is a region map (nearest), anything else Grad-CAM (bilinear). Bytes are rounded half-up
  (`floor(x · 255 + 0.5)`) so every renderer agrees.
- The `dev` extra installs matplotlib, so CI runs the `plot_explanation` tests instead of skipping
  them. matplotlib stays optional for users.

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
