# Changelog

## 0.9.2 — 2026-11-08

A 0.0.1 bump from 0.9.1. The calls are unchanged. Structure documents may include `verification`.

## 0.9.1 — 2026-11-08

Needs the API release with password reset: deploy that first.

### Added

- `auth.forgot_password` and `auth.reset_password` request a reset link and set a new password from it.

## 0.9.0 — 2026-11-08

Needs the API release with outcomes and the `performance` analysis: deploy that first.

### Added

- **Outcomes and performance over time.** `predictions.create(actual=...)` logs what really happened with the
  prediction; `predictions.log_actual(id, actual)` and `predictions.log_actuals(...)` attach it later, from a dict,
  a list of `{"prediction_id", "actual"}` items or a pandas Series indexed by prediction id. Batches of any size are
  sent 1,000 at a time; NaN and missing values are skipped, and an explicit `None` in a list clears an outcome.
  A classifier's outcome is its class name, its position, or `True`/`False` for a binary model; a regression's is a
  number.
- `models.performance(model, window="week", days=90)` measures the model against outcomes: AUC, accuracy, precision,
  recall, F1, Brier score and calibration for binary models; accuracy, macro F1, per-class numbers and a confusion
  matrix for multiclass; MAE, RMSE, bias and R² for regressions; overall and per day, week or month.
  `display.rows` turns its series into a table.
- `analyses.create(type="performance")` compares the newest outcomes with the ones before them (or with the
  labelled reference rows, `options={"baseline": "reference_data"}`) and raises a `high` insight when the drop is
  larger than chance explains. Full analyses include it.
- `predictions.list(has_actual=True)` lists only predictions with an outcome. Notebook cards show the outcome and the
  performance insight.

## 0.8.0 — 2026-11-07

Needs the API release that compiles `opaque` layers: deploy that first.

### Changed

- **Every PyTorch and Keras model uploads and is drawn.** A layer or operation Kaimei cannot replay (attention,
  `matmul`, a reshape to another rank, a channel shuffle, a module of your own) becomes an `opaque` step in the spec:
  the framework's own name for it, its attributes, its output shape and its tensors, which the dashboard draws and
  lists like any other layer. The model is then drawn, not run, and its summary says which steps and why. Before,
  such models were refused with `unsupported_layers`; `on_unsupported="error"` on `upload`, `upload_structure`,
  `extract_structure` and `estimator_to_spec` keeps that behaviour.
- Dense PyTorch models that `torch.fx` can trace keep their op-by-op reading even when a step is drawn, so a
  residual add or a functional activation next to it still runs the way the code says.
- `kaimei.structure.drawn_steps(spec)` lists the steps of a spec that are drawn but not replayed.
- **Every spec says where it came from.** `metadata` on the spec records the model's class, the framework and
  its version, the Python version, this client's version and when it was read; `extract_structure(metadata=...)`
  and `upload_structure(metadata=...)` add your own entries. The API keeps it on the structure document, and the
  dashboard shows it under Details. (`upload(metadata=...)` is still the model's own metadata.)
- Parameters in the structure document carry their stored precision (`dtype`: `f16`, `f32` or `f64`).

## 0.7.2 — 2026-11-07

Needs the API release that understands flat forests and composites: deploy that first.

### Changed

- **Uploads are checked more strictly, and some models are drawn instead of run.** A model is sent with a few rows and
  its own library's answers to them, and the API scores those rows with the engine that will run it. If they don't
  match, the model is drawn, not run. The rows aren't stored, and `weights="summary"` doesn't send them. That covers
  PyTorch and Keras models, convolutional ones included. `SVC` and `NuSVC` with `probability=True`, whose labels can
  disagree with their probabilities, and any subclass with its own `predict` are drawn too. **Models already uploaded
  keep their old label until they are re-uploaded**, because the server never holds the model itself to check it.
- Dense PyTorch networks are read op by op, as convolutional ones already were, so `F.relu`, `torch.sigmoid` and a skip
  connection from the inputs run correctly rather than being missed. Modules `torch.fx` can't trace still go through
  forward hooks and are drawn.
- `nn.GELU()`, `F.gelu` and Keras `gelu` are drawn, not run: they default to the exact form, and Kaimei runs the tanh
  form, which is too close for a check on convolutional models to tell apart. `approximate="tanh"` runs.
- Reference rows go up in pieces of about 32 MiB, so a large `X` no longer hits the API's limit on one request.
- `upload(X=...)` also checks the model on your rows, not only on rows around zero and its splits.
- Summaries and cards say what a base value is, and call a model's outputs scores when it has no probabilities.
- A plot of a very wide model has a row for what wasn't attributed to one input, so its last bar is the prediction.
- `upload` warns when reference rows were refused, instead of leaving them out silently.

### Fixed

- **Models could upload as something they aren't, and report wrong numbers as exact.** The extractor
  walked into fitted children and kept the first one that looked exact. A forest over 30 trees or
  4,000 nodes, sklearn's default `RandomForestClassifier()` included, ran as its first tree. A
  `VotingClassifier`, `BaggingClassifier` or a vendor's list of models ran as its first member, and
  `CalibratedClassifierCV` and `TransformedTargetRegressor` ran without the step they add. **Re-upload
  any model affected by this**: a stored spec keeps its old numbers.
- AdaBoost, and bagging over a subset of columns, no longer run as a plain average of their trees.
- Scaler pipelines apply every scaler as sklearn does. Only the last of several scalers was kept, and
  `StandardScaler(with_mean=False)` or `with_std=False` used the wrong centre or scale.
- Single trees and small forests compare inputs in single precision as sklearn does, which decides a
  row that sits exactly on a split.
- `XGBRegressor(objective="reg:logistic")` and `LGBMRegressor(objective="cross_entropy")` reported one
  minus their output. Bare XGBoost and LightGBM boosters now carry class labels and go through the check.
- `XGBClassifier(missing=0)`, or any other marker for missing values, now takes the branch XGBoost does
  for that value. Before, it scored those rows as ordinary numbers.
- A `CatBoostRegressor` no longer uploads as a multiclass classifier, whichever way its trees grow.
- A PyTorch layer called twice no longer fails with `Duplicate structure node`.
- `predictions.retrieve` encodes the id, so one holding `?` or `#` reads back.
- Models fitted on data with missing values can be uploaded. sklearn gives them infinite thresholds, which
  JSON can't carry, and those thresholds also made the check skip them.
- A model with several regression outputs is drawn but not run. It used to report its first output as the
  model's only output, as exact.
- `upload(X=...)`, or any call whose body held `NaN`, raised a raw `ValueError`. `NaN` now travels as `null`,
  which is how JSON writes a missing value.
- `predict`, `explanations.create` and `trace_structure` raise `ValueError` for a `NaN` on a feature that can't take
  a gap, after one extra request to read the model. As `null` it would have used the feature's baseline.
  Infinity in a request now says what's wrong.
- **`weights="summary"` sent real numbers.** Trees, small forests, starting scores, per-tree weights, calibration
  curves and scaler statistics all went up as they were, and the statistics of a small tensor gave away its values.
  Now a summary holds shapes, feature importances or how often each feature is split on, and statistics only for
  layers of 256 values or more. Those statistics still include a layer's largest and smallest value.
- A pipeline's `SimpleImputer` is no longer read as handling gaps, which was wrong for some rows. Its fill values
  become the features' baselines instead, so a `null` gets what the imputer would give it, and reference rows with gaps
  are kept. A `SimpleImputer(missing_values=-999)` raises `StructureError`: it replaces real numbers, so reading it as
  nothing was wrong for those rows, and the check didn't try them.
- PyTorch and Keras models ran as exact with wrong numbers when `forward()` held logic that wasn't read. They now run
  correctly when it can be read, and are drawn when it can't.
- A PyTorch classifier that ends in logits gets its sigmoid or softmax, as image models already did.
- A scaler in front of any tree model is exact on rows sitting on a split. A threshold is now moved into the input's
  units by searching with the scaler's own arithmetic and the library's own comparison, not by multiplying it back out.
  This also lets a scaler precede XGBoost, LightGBM and CatBoost.
- Multiclass `SGDClassifier(loss="log_loss")` and `OneVsRestClassifier` over anything with a margin (logistic regression,
  `LinearSVC`, `RidgeClassifier`, `Perceptron`, `LinearDiscriminantAnalysis`, a linear `SVC`, or those behind a scaler)
  decide by their margins, as scikit-learn does, where two classes' probabilities round to the same number.
- A model only counts as matching if it was actually compared. One whose `predict` needs a table is now asked with one,
  and one that refuses to answer at all is drawn, not run. It used to be called exact without being compared.
- A subclass with its own `predict`, or your own class, inside a vote, a bag or a stack is drawn, not run. Only the
  top of a model was looked at before.
- A hard vote or `OneVsRestClassifier` over `SVC(probability=True)` is drawn, not run: its members decide by their
  margins, and what's read of them is their probabilities.
- An imputer behind a scaler gives a `null` the fill value in the input's units, not the scaler's.
- A scaler in front of a stack with `passthrough=True` also reaches the features it passes through.
- XGBoost's `missing` marker behind a scaler is drawn, not run, because no one input number stands for it.
- A classifier with several outputs of different sizes is drawn, not run. It was read as two classes with odd names.
- A `PoissonRegressor` and the other log-link models run where their library overflows to infinity.
- Members that don't fit together, such as a multiclass `SVC` in a bag or a vote, are drawn. `upload` raised a raw
  `ValueError`.
- A pipeline inside a pipeline's preprocessing is read, and two scalers in a row no longer clash on a one-feature model.
- `TransformedTargetRegressor` runs with `MinMaxScaler` and `RobustScaler` as well as `StandardScaler`.
- A two-output PyTorch or Keras classifier with `task="binary_classification"`, and a Keras `Sequential` that ends in
  logits, run. So do Keras models with a softmax after `Add` or `Concatenate`, or dropout after their softmax.
- A layer or a reader's own step can share its name with a feature. `upload` used to refuse the model.
- LightGBM fitted on a table with spaces in its column names keeps them, and `X` is read before anything is created, so
  an `X` that doesn't fit leaves no model behind.
- `feature_names` that moves any name the estimator was fitted on is refused, even if others are new.
- A PyTorch model whose `forward()` feeds its layers in a way that can't be read gets a clear `unsupported_layers` error,
  not the API's complaint about a weight's shape.
- A Keras model ending in `Dense(n)` then `Softmax()` or `Activation("softmax")` is read as a classifier. It was refused.
- A skip connection from a Keras network's raw inputs runs. It was refused with an unhelpful message.
- CatBoost, LightGBM and XGBoost models fitted on a table keep its column names when `feature_names` isn't passed. The
  names a library makes up for a bare array (`0`, `Column_0`) are no longer taken for real ones, so `X`'s columns name
  the model.
- A multiclass `SVC` is drawn, not run. It votes across pairs of classes, which a linear reading can't do.
- `feature_names` in a different order from the estimator's `feature_names_in_` raises `StructureError`.
- The check against the estimator scales with the size of the outputs. Outputs around `1e-7` used to match anything.
- A `LightGBM` or `XGBoost` subclass keeps its class labels and its checks. The wrong number of `feature_names` is
  a `StructureError` for every kind of model. For trees it used to be accepted, or a raw `IndexError`.

### Added

- Tree models of any size now run: forests, gradient boosting, histogram boosting, AdaBoost, bagging,
  XGBoost, LightGBM and CatBoost. Large ones travel as flat binary arrays in full precision.
- Models made of other models now run: voting (soft, hard, weighted), stacking, calibration, bagging,
  one-vs-rest, and `TransformedTargetRegressor` with a scaler or a NumPy inverse.
- `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` with a log link, a binary linear
  `SVC(probability=True)` and multiclass `SGDClassifier(loss="log_loss")` run exactly.
- `RobustScaler`, `MaxAbsScaler`, any `MinMaxScaler` range and `SimpleImputer` work in pipelines, and a
  scaler can precede any tree model.
- Every scikit-learn, XGBoost, LightGBM and CatBoost upload is checked against the estimator's own output,
  including rows exactly on its splits and rows holding zeros. A spec that doesn't reproduce it is drawn but
  not run, with a warning. Keras and PyTorch models are scored against their own outputs too.
- XGBoost's `dart` booster runs, and AdaBoost runs inside a calibrator or a stack.
- **Rows with missing values run, on models that handle them.** The check also scores rows with gaps. Where the
  library accepts them and the reading matches, numeric features get `allow_missing` and a row can send `null`.
  Shapley methods explain such a row; LIME and counterfactuals return `missing_value_not_supported`.
- `LinearSVC`, `RidgeClassifier` and others without `predict_proba` are marked `uncalibrated`, because
  their 0..1 outputs are scores rather than probabilities.

### Changed

- `client.models.upload_tensors` sends double-precision arrays as `f64` unless you pass `dtype`.

### Changed in the API this release needs

These change results for models you already have, and they don't depend on the client version:

- **Rows with a missing value now count.** Analyses and reference averages used to leave them out, and KernelSHAP
  explained a missing value as if it were present. Stored insights and explanations of those models may differ.
- `inference.url` can't be a loopback, private-network or cloud metadata address (`invalid_inference_url`), and moving
  it to another host needs its credentials sent again (`inference_credentials_required`).
- A reply from an endpoint is limited to 64 MB and to `timeout_s` in all, and an organization can have 12 calls to
  endpoints in progress at once (`429 inference_busy`). Analyses wait for a free call instead of failing.
- An organization's analyses take turns, leaving a worker for other organizations, and a job still in line isn't
  reported as lost.
- Replacing a model's structure updates `allow_missing` to what the new structure does. Class labels follow the order
  of the outputs. A `target_class` analysis scores accuracy for that class, and rows with no label no longer count.
- A binary `target_class` analysis now names that class. It used to name the positive class while reporting the other
  class's share.
- A forest with no rule for missing values refuses a missing value (`missing_value_not_supported`), even when a feature
  declares `allow_missing`.
- A link from outside the dashboard asks before signing you in, and a confirmation link works once.
- JSON nested more than 64 levels deep is refused (`too_deeply_nested`), an image feature can hold at most 8,000,000
  values, and an address over 254 characters isn't an email. A `prediction_id` holding a slash can now be read back.
- An email address that signed up with a password and never confirmed it loses that password when its real owner
  signs in with Google or GitHub. Otherwise whoever signed up first would get in to the owner's account.
- **A multiclass explanation with no `target` now explains the predicted class.** It used to explain the
  first class while naming the predicted one, so stored explanations of those models may differ.
- A multiclass analysis says which class it covers, and takes `options.target_class`.
- Explaining a row with nothing to compare it with returns `no_reference_population`. It used to return
  all-zero attributions.
- `NaN` or infinity as an input is refused. It used to return a prediction, often of 0.
- An output that isn't a finite number returns `non_finite_output`. It used to be reported as 0.
- A network with several regression outputs also returns `values`, all of them. `value` is still the first,
  and explanations explain that one.
- Analyses run in a small pool of their own, so a burst waits in line (`status: queued`) rather than making other
  requests fail. A job lost to a restart reports `failed` after an hour rather than `running` forever. A busy
  database returns `503 database_busy` with `Retry-After`.

## 0.7.1 — 2026-11-01

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

- README: an Examples section (the [Titanic example](https://github.com/KaimeiAI/titanic-kaimei-example), the scikit-learn and image examples); links to [kaimei.ai](https://kaimei.ai) and the [API reference](https://kaimei.ai/docs) instead of a repository that isn't public.
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
