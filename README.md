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

The same call as `POST https://api.kaimei.ai/v1/explanations`. See [kaimei.ai](https://kaimei.ai) for the language-agnostic REST examples and the [API reference](https://kaimei.ai/docs/api.html) for every endpoint.

## Documentation
Our documentation page can be found here: [`Kaimei Docs`](https://kaimei.ai/docs/)

To get started, see the [`Quickstart`](https://kaimei.ai/docs/quickstart) page, and reference the examples below.

## Install

```bash
pip install kaimei
```

To extract and upload a fitted estimator you also need NumPy (and your training stack):

```bash
pip install "kaimei[models]"
```

A model Kaimei has no reader of its own for is uploaded through ONNX ([below](#any-model-through-onnx)):

```bash
pip install "kaimei[onnx]"   # NumPy, onnx and onnxscript; a PyTorch model needs nothing more
pip install skl2onnx         # for scikit-learn estimators
```

For a self-hosted or local Kaimei API, pass its address:

```python
client = Kaimei(api_key="km_test_...", base_url="http://localhost:8000")
```

`KAIMEI_API_KEY` and `KAIMEI_BASE_URL` are read from the environment when arguments are omitted.

Create an API key in the [dashboard](https://dashboard.kaimei.ai) (session login). Keys cannot create other keys.

## Examples for text and image models

Two short runnable examples of models that are not tables are in [`examples/`](examples/README.md): a scikit-learn
text classifier and a PyTorch image segmenter, each as a script and as a notebook.

## Examples

- **[titanic-kaimei-example](https://github.com/KaimeiAI/titanic-kaimei-example)**: a complete workflow on the Titanic data, in five notebooks with their outputs (and Colab badges):
  - upload a scikit-learn model;
  - explain passengers, with counterfactuals and two models compared;
  - analyse the whole model;
  - monitor production traffic and catch drift;
  - ship an improved version;
  - explain a model through your own infrastructure: XGBoost here, via TreeSHAP values and a scoring endpoint. Kaimei can also run XGBoost itself.
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
- **Explain / predict / analyse** — when the spec is executable: see the table below. The API runs that structure in-process.
- **Otherwise** — pass a live scoring URL, run an [agent](#models-that-stay-with-you-the-agent) beside the model, or POST pre-computed attributions:

```python
client.models.upload(
    my_custom_model,
    name="Churn scorer",
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

- `externalize_above=` — values per tensor above which it uploads as binary (default 65,536; `None` keeps everything inline). A single value, or a value with no axes, is never lifted: `externalize_above=0` lifts every other constant.
- `weights="summary"` — upload shapes and statistics, not the weights. The architecture is drawable. Layers under 256 values send no statistics, since those would give the values away. Trees and models made of other models send only how many trees there are and how often each feature is split on, or the model's feature importances. Explain / predict then need `inference={"type": "remote", ...}` or an agent. A model of any size can be registered this way: no tensor's size and no parameter count is limited for weights that are not sent. A value a graph's input is always given (`inputs={name: {"value": ...}}`) is kept at home too, unless it is whole numbers (up to 16) or a single number; such a summary is drawn, not run.
- `weights="pictures"` — a summary that also sends a small picture of each float tensor of 256 values or more, for the dashboard to draw where the weights would be: at most 64 by 64 cells, each the strongest value of its block, at 8 bits. You ask for this; a summary alone never sends it. A large tensor is reduced to almost nothing by it. One of 64 by 64 values or fewer is all there, at that precision, so do not ask for pictures of a model whose small layers must stay private.
- `extract_structure(estimator, ...)` returns the spec and the tensors without uploading, for inspection or a custom upload.
- `client.models.tensors(model)` lists what was uploaded; `structure_parameter_values(model, "fc1.weight")` reads a tensor back as float32.

A runnable walkthrough lives in [`examples/upload_sklearn.py`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/upload_sklearn.py) and [`examples/upload_sklearn.ipynb`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/examples/upload_sklearn.ipynb).

### Supported estimators

| Estimator | What is uploaded | Executable for explain / predict |
| --- | --- | --- |
| sklearn linear (`coef_` / `intercept_`) | Exact weights | Yes |
| sklearn `MLPClassifier` / `MLPRegressor` | Exact layers | Yes |
| sklearn tree | Full node list, or flat binary arrays above 2,000 nodes | Yes |
| sklearn forest / extra trees, any size | Every tree: a node list when small, flat binary arrays above that | Yes |
| sklearn `GradientBoosting*`, `HistGradientBoosting*`, `AdaBoost*` | Every tree, and how they combine | Yes, unless a split is categorical |
| XGBoost `gbtree` and `dart`, LightGBM, CatBoost (sklearn API or booster) | Every tree, and how they combine | Yes, unless a split is categorical or the reading fails its check |
| sklearn `Bagging*`, `Voting*`, `Stacking*`, `CalibratedClassifierCV`, `OneVsRestClassifier` | Every member, and how they combine | Yes, when every member is |
| `TransformedTargetRegressor` with a scaler or a NumPy inverse (`expm1`, `exp`, ...) | The regressor and the inverse | Yes. Your own Python function can't be stored: pass `inference` remote |
| sklearn `Pipeline` + `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, one or several, and `SimpleImputer` with its default `missing_values=NaN` | Scalers folded or inserted, then the estimator | Same as the final step |
| Keras / TensorFlow `Dense`, `Conv2D` (any stride, padding, dilation, groups), `DepthwiseConv2D`, `SeparableConv2D`, `BatchNormalization`, max / average / global pooling, `Add`, `Multiply`, `Concatenate`, `ZeroPadding2D`, `Rescaling`, `Normalization`, flatten, dropout, LayerNorm; functional graphs and nested base models | Layer weights and the layer graph | Yes, unless a layer is unrecognized (`approximate`) |
| `torch.nn` CNNs: `Conv2d` (stride, padding, dilation, groups), `BatchNorm2d`, max / average / adaptive pooling, residual `+`, `torch.cat`, squeeze-excitation `*`, `Flatten`, `ReLU6` / `Hardswish` / `SiLU` and the other common activations (ResNet, MobileNet, EfficientNet, DenseNet, VGG, SqueezeNet) | Traced with `torch.fx`, so functional ops are kept | Yes; pass `input_shape=(C, H, W)` |
| `torch.nn` `Linear`, common activations, Dropout, LayerNorm, Embedding, BatchNorm1d | Traced op by op, including `F.relu` and skip connections, or in forward-hook order when it can't be traced | Yes, when traced and it gives the module's own outputs. Otherwise drawn (`approximate`) |
| Wrapper around any of the above that predicts as it does (`model_`, `estimator_`, `best_estimator_`, `module`, …) | Same as the inner estimator | Same as the inner estimator |
| scikit-learn estimators with no reader of their own: `SVC`, `NuSVC` and `SVR` with any kernel, with or without `probability=True`; k-nearest neighbours; naive Bayes; `QuadraticDiscriminantAnalysis`; `GaussianProcessRegressor`; `IsolationForest`; `OneVsOneClassifier`; pipelines with `PCA`, `PolynomialFeatures` or an imputer before such an estimator | Converted to ONNX with `skl2onnx`: every operator and tensor | Yes, when the conversion gives the estimator's own outputs and its own class, which is checked. Needs `skl2onnx`. An SVC with `probability=True` is read by its probabilities when their most probable class is the class `predict()` gives on every check row (scikit-learn documents that the two can differ on other rows: the upload warns and the structure says so in `metadata.predicted_class`), and by its `decision_function` otherwise |
| `torch.nn` recurrent layers (`LSTM`, `GRU`), `nn.TransformerEncoder`, `MultiheadAttention`, `Conv1d`, `Upsample`, exact GELU, and modules that take token ids or several inputs | Converted to ONNX with PyTorch's own exporter | Yes, when the conversion gives the module's own outputs. Pass `sample=`. Needs `onnx` |
| An ONNX model: a `.onnx` path, its bytes, or an `onnx.ModelProto` | Every operator and tensor | Yes, for the operators the API runs. Others are drawn, and the upload names them |
| `SVC` and `NuSVC` with `probability=True` when `skl2onnx` is not installed, and a hard vote or `OneVsRestClassifier` of them unless it converts through ONNX | Drawn, not run: `predict` goes by the margin and the probabilities by a separate fit, so a label and its probability can disagree | No |
| A subclass with its own `predict`, or your own class inside a vote, a bag or a stack | Drawn, not run: its learned numbers don't show what it does | No |
| `LinearSVC`, `RidgeClassifier` and others without `predict_proba` | Exact weights, marked `uncalibrated`: labels match sklearn, and the 0..1 outputs are scores rather than probabilities | Yes |
| Estimators no converter takes (your own classes, pipelines with steps `skl2onnx` does not know), and the other estimators of the scikit-learn ONNX row above when `skl2onnx` is not installed | Refused with an explanation, which says what to install when that would help | No — register with `client.models.create` and an `inference` URL or an agent |
| Anything else with `feature_importances_` | Importance summary, unless it converts through ONNX | No — pass `inference` remote or attributions |
| A model too large for the API to run, or whose weights must stay with you | Shapes and statistics (`weights="summary"`) | By its [agent](#models-that-stay-with-you-the-agent) |

Every scikit-learn, XGBoost, LightGBM and CatBoost upload is scored against the estimator on rows around and exactly on its splits. A reading that doesn't reproduce it is marked `approximate`, so it is drawn but not run. So far that covers `GradientBoosting*(init=<an estimator>)`, LightGBM `linear_tree` and CatBoost `MultiClassOneVsAll`. A model with several regression outputs is also drawn but not run, because a prediction reports one number or one class.

The API repeats the check when the model arrives, for PyTorch and Keras models too, with the engine that will run it. A few rows and your library's answers to them go up with the model and aren't stored. `weights="summary"` doesn't send them. Kaimei's own readers run GELU in its tanh form, so they only draw the default exact form (`nn.GELU()`, `F.gelu`, Keras `gelu`). A PyTorch model with it is converted through ONNX and run as it is when `onnx` is installed; otherwise pass `approximate="tanh"`.

**Missing values.** The same check scores rows with gaps. If the library accepts them and the reading matches, the model's numeric features are registered with `allow_missing`, and a row can send `null`, or `NaN` from pandas, for a value it doesn't have. That covers XGBoost, LightGBM, CatBoost, `HistGradientBoosting*`, and scikit-learn trees and forests that support missing values. Other models need every value: a `null` takes the feature's `baseline` if it has one, and is refused otherwise, while `NaN` raises `ValueError`, since it means the value is missing rather than that you want the baseline. A pipeline's `SimpleImputer` is left out of the reading, so that model takes numbers only. A pipeline read through its `ColumnTransformer` (or an encoder at its front) is different: each column with an imputer has the imputer's fill as its `baseline`, so a `null` for it is what the pipeline itself does with a gap. Shapley methods explain a row with a gap; LIME and counterfactuals need every value. Reference rows with a gap are stored but sit out of averages, histograms and comparison populations.

Layers and operations Kaimei's own readers cannot replay (attention, `matmul`, a reshape to another rank, ShuffleNet's channel shuffle, a module of your own) send a PyTorch model through ONNX when `onnx` is installed, and it usually runs: see the next section. Without `onnx`, with `via="native"`, or when the conversion does not give the model's outputs, they become `opaque` steps: the spec carries the framework's own name for the step, its attributes, its output shape and its tensors, and the dashboard draws and lists them like any other layer. The model is then marked `approximate`: it visualizes, with a "drawn, not run" badge naming how many steps, but cannot be queried unless you add a remote URL or an agent. Pass `on_unsupported="error"` to have such models refused with `StructureError(code="unsupported_layers")` naming the steps instead. It refuses any other reading Kaimei would only draw as well, on every route: one that does not give the model's own outputs or its own class, or a summary of feature importances. (`weights="summary"` is not refused for it.)

### Any model, through ONNX

Kaimei's own readers know a fixed set of estimators and layers. A model they cannot read is converted to ONNX in memory, on your machine, by the framework's own converter, and uploaded as a graph of ONNX operators that the API runs one by one. The ONNX file is not uploaded and not stored: what goes up is the structure and the tensors, as for every other model.

Graphs, and the text, generation and agent features below, need Kaimei API 1.3.0 or later; `client.health().version` says which API you have. An older API refuses a graph and nothing is made. With `via="auto"` the client then uploads the model as Kaimei's own readers read it, as clients before 0.12.0 did (refused, or drawn and not run), and warns. `via="onnx"`, an ONNX model given directly and `upload_structure` raise an error naming API 1.3.0. `upload_tensors` sends an array of whole numbers or flags in its own type, which an older API refuses too: pass `dtype="f32"` for one.

```python
client.models.upload(knn, name="Neighbours", feature_names=names, X=X_train, y=y_train)      # scikit-learn, through skl2onnx
client.models.upload(forecaster, name="Demand", task="regression", sample=window)             # a torch LSTM; window is a (1, 24, 3) tensor
client.models.upload("model.onnx", name="From a file", feature_names=names)                   # an ONNX model itself
```

- `via="auto"` is the default: Kaimei's own readers first, and ONNX when they refuse the model, can only draw it, or cannot call it (token ids, several inputs). `via="onnx"` and `via="native"` force a route. The structure's `metadata.route` says which was taken. An ONNX model is read as a graph whatever `via` says. `inputs=`, `tokenizer=`, `mask_layout=` and an `output=` other than a module's first are read only from a model converted to ONNX: with any of them `via="auto"` converts it first and refuses it rather than leave them out, and `via="native"` refuses them. Every argument is checked before the model is read, on every route, and a mistake names the argument (`StructureError.param`).
- `sample=` is an example input, as you would call the model with it: a tensor with its batch axis, or a tuple or dict of them for several inputs. Token ids and several inputs need it. Without it a table of `feature_names` or an image of `input_shape` is assumed. For a scikit-learn pipeline whose `ColumnTransformer` picks columns by name, `sample` is a table that holds those columns (other columns may be there or not).
- `inputs=` says what an input is when its name and shape do not: `{"pixels": {"feature": "photo", "layout": "nchw"}}`, `{"ids": {"feature": "review", "part": "ids"}}`. A row sent to the model is then `{feature: value}`, with a series or a tensor as nested lists. It names inputs of the model as converted: `X` for scikit-learn, the input names of a Keras model's signature, a PyTorch `forward`'s parameter names (`input_0`, `input_1`, ... for a module whose `forward` takes more values than it names, `forward(self, x, *rest)`), and `image` for a module wrapped by `preprocess=`, which also accepts the module's own name. A name that is none of them is an error.
- `output=` picks the output the prediction is read from when the model has several, and `labels=` names the classes: a list, a tuple, a NumPy array or a pandas Index (`labels=encoder.classes_`). A classifier that gives one value for a row (a margin, one sigmoid) takes two names, negative first. `max_length=` and `output=` take any whole number, NumPy's included.
- `task` may also be `text_generation` (a language model), `generation` (the answer is a tensor: a picture, a series) or `embedding`.

**The check holds the model's outputs, not the conversion's.** A few rows go up with the structure with what the original library (PyTorch, scikit-learn) gives for them, and the API repeats them with its own engine before it runs the model. A conversion that is further than 1e-4 of its outputs' size from the library is marked `approximate` and is drawn, not run. An ONNX file given directly is checked against ONNX Runtime, or ONNX's reference evaluator when ONNX Runtime is not installed. A model that draws at random (at its top, in a branch or a loop, or in a function the file defines) is compared with the ONNX model run with the seeds Kaimei gives its random steps, not with the original, which draws differently each run: `metadata["compared_with_original"]` is `false` and `metadata["checked_against"]` ends ", with fixed seeds".

**A classifier is exact only when Kaimei reproduces its scores and its class.** On every row of the check, the class Kaimei reports (the highest probability; for a yes-or-no model, the positive class at its threshold) is compared with the model's own `predict`. Where they differ, Kaimei reads the model the way that agrees:

- `TunedThresholdClassifierCV` and `FixedThresholdClassifier` are uploaded with their own decision threshold (`outputs.threshold`; `outputs.positive_label` when the positive class is the first one). A threshold on a decision score is carried squashed, as the score Kaimei reports is. A threshold that cannot be carried (one outside 0..1 on a probability, one beyond ±30 on a decision score) leaves the model drawn, with the reason. The Kaimei API registers the model with the decision its structure carries (the threshold, and the class it is for), unless `outputs` gives another (then Kaimei reports by yours, and `models.upload` warns that it is not the model's own); after `models.upload_structure` the API makes an existing model follow the structure's threshold when the model has none of its own (one half), and a model registered with another threshold keeps it, with a warning.
- A classifier whose `predict` goes by its scores where its probabilities tie (`SGDClassifier(loss="modified_huber")`) is read by its `decision_function`: its shares are those scores squashed, not its probabilities.

When no reading agrees on every row, the model is uploaded **drawn, not run**; its summary says on how many of the rows it was checked on the class differs, and why when that is known. `on_unsupported="error"` refuses it. Two kinds of model keep a note on an upload whose check rows all agree, because their rule is known to differ on other rows: an SVC or NuSVC with `probability=True` (its `predict` goes by the margin), and a multiclass SVC or NuSVC without `break_ties=True` (its `predict` counts one-vs-one votes). Pass rows you care about as `X` (they are checked), or set `break_ties=True` (it needs no refit). The note is in `metadata["predicted_class"]`, and the upload warns. An SVC or NuSVC with `probability=True` is read by its probabilities when the most probable class is the class `predict()` gives on every check row; where a check row disagrees, it is read by its `decision_function` instead, whose class is the model's own (for more than two classes, with the note about one-vs-one votes unless `break_ties=True`). An SVC with `decision_function_shape="ovo"` and more than two classes is refused: set it to `"ovr"`.

**Precision.** A scikit-learn model read through ONNX is computed in single precision when that gives the model's own numbers on rows from the range of its data, and in double precision otherwise (`metadata.model_precision` says which). The check holds your rows (`X=` on `upload`, `check_rows=` on `extract_structure`) as the doubles they are, and rows made up where the data lies: from the statistics of the pipeline's first scaler, from your rows, or from the points the model keeps (nearest neighbours, naive Bayes means). Give rows when a column holds large values that differ by little (a time in seconds, an amount in cents): without rows or a scaler nothing says where the values lie, the summary says so, and a model that tells doubles apart is computed in doubles to be safe.

For this release the route was run against the API on 33 models, and every prediction compared with the library's own on 20 held-out rows:

- scikit-learn `GaussianNB`, `MultinomialNB`, `KNeighborsClassifier`, `KNeighborsRegressor`, `SVC` (RBF with probabilities, polynomial without, two and three classes), `SVR`, `NuSVC`, `QuadraticDiscriminantAnalysis`, `GaussianProcessRegressor`, `IsolationForest`, `OneVsOneClassifier(LinearSVC)`, and pipelines with `PCA`, with `PolynomialFeatures`, and with `SimpleImputer` and `StandardScaler` before an `SVC`;
- PyTorch LSTM, GRU and two-layer bidirectional LSTM models, an `nn.TransformerEncoder` text classifier, a classifier that takes token ids and an attention mask, a small GPT, `Conv1d`, `Upsample`, a model with two inputs, two autoencoders and an embedding model;
- three ONNX files: an MLP, boosted trees with a `ZipMap` output, and a small CNN.

All of them ran. The largest difference was 3e-5 for scikit-learn (on regression values) and 2e-7 for PyTorch.

What it does not do:

- A graph's shapes are fixed when the model is converted: one row at a time, with the sizes of `sample`.
- The API runs each operator in NumPy. That is fine for small and medium models and slow for large ones: a large model is better run by its agent.
- An operator the API has no kernel for makes the model drawn, not run, and the upload warns with its name. In a run of 75 small PyTorch modules, each built around a few operators, 71 ran. Of the other four, a graph whose tensor sizes depend on the data (`Unique`) is drawn, and `torch.cdist` converts to ONNX 1.3e-4 away from PyTorch and is drawn for that. `torch.linalg.inv` has no ONNX operator, and an LSTM with `proj_size` is not read: neither is uploaded this way.
- A classifier without `predict_proba` (an `SVC` without `probability=True`, a `OneVsOneClassifier` of `LinearSVC`) is read through its `decision_function`. The shares shown are those scores squashed to add up to one, not calibrated probabilities.
- A scikit-learn classifier with several outputs of its own (a k-NN, a tree or a forest fitted on a table of answers, a `MultiOutputClassifier`, a `ClassifierChain`, whose `classes_` is a list) is not read through ONNX: the refusal says so. A classifier fitted on a table of yes-or-no answers that gives one probability for each (`OneVsRestClassifier`, `MLPClassifier`, `RidgeClassifier`) is read, as `multilabel_classification`.
- A scikit-learn tree that learned where missing values go cannot be converted by `skl2onnx`. It is refused with that reason wherever it sits (in a calibration, behind a target transform, in a stack or a vote); Kaimei's own readers (`via="auto"`, `via="native"`) read such trees.
- Keras and TensorFlow models go through `tf2onnx` when it is installed: Keras 3 models (recurrent layers included)
  through `tf.function` with the model's own input signature, older ones through `tf2onnx.convert.from_keras`.
  Without `tf2onnx` (or when its conversion cannot be used), a Keras model whose input is not a table or a picture
  is uploaded as Kaimei's own readers draw it (drawn, not run), as clients before 0.12.0 did, with a warning.

### What Kaimei works out by itself

Most uploads need the model and an example input, nothing more. The structure says what the model is:

- **The task.** Outputs that are probabilities make a classifier, and so do `labels=` with a name for each value (the
  logits a PyTorch classifier ends in) or a single value out of a sigmoid; a sigmoid for each of several labels is a
  multi-label classifier. A Hugging Face model says it in its class and config: `...ForSequenceClassification`,
  `...ForImageClassification` and `...ForAudioClassification` are classifiers (classes from `id2label`, several at
  once when `problem_type` says so), `...ForCausalLM` writes text, `...ForConditionalGeneration` is an encoder-decoder,
  `...ForSemanticSegmentation` gives masks, `...ForObjectDetection` boxes, and a bare `...Model` with a pooled output
  an embedding. A scikit-learn clusterer is a clustering and a `diffusers` pipeline a generator with its scheduler.
  Anything else that answers with several numbers stays a regression, which is drawn: the upload's warning says
  which `task=` to pass.
- **The inputs.** Every input of the model is a feature of its own. Whole numbers called `input_ids`, `ids` or
  `tokens`, or the only run of them that is looked up in an embedding table, are a text (`attention_mask`,
  `token_type_ids` and positions beside them are worked out from it); any other whole number (a user, an item, a
  category) is an integer column, ranged by the table it is looked up in. A single float is a column (a diffusion
  model's timestep), a run of up to 1,024 a table, and a longer one a tensor. A run of at least 256 called
  `input_values`, `waveform`, `audio`, `speech` or `signal` (and the like) is a recording (`type: "audio"`, at 16 kHz
  unless `inputs=` gives `sample_rate`). Two axes are a series, three with 1, 3 or 4 channels a picture. `inputs=`
  still says otherwise for any of them.
- **The example has to be one Kaimei can give.** When the example input, turned into the features and back, is not
  what it was (two inputs read from one feature, say), the upload says which and asks for `inputs=`. Labels of the
  wrong count are refused.
- A Hugging Face language model is exported with its cache turned off (`use_cache`), so it converts as it is. A model
  only a conversion can read, which does not convert, is refused with the reason rather than uploaded as a list of
  its layers.

### More frameworks, and pipelines read exactly

- A **TensorFlow SavedModel** (its directory) and a **`.tflite`** file are read through `tf2onnx` and checked against
  TensorFlow. A **JAX** function, jitted function or Flax NNX module goes through `jax2onnx` (`framework="jax"` for a
  plain function; `sample=` gives the shapes, and the model is converted for one row, whatever the number of rows
  the sample has). A **`.gguf`** file is opened by Hugging Face `transformers` (with `gguf` and `accelerate`
  installed) and read as the PyTorch model it is.
- A scikit-learn **bag-of-words** pipeline (`TfidfVectorizer`, or `CountVectorizer` and `TfidfTransformer`, then a
  model) is read exactly, with its text split as scikit-learn splits it, so the feature takes text:
  `client.models.upload(pipeline, name=..., X=texts)`, where `texts` is a list, a pandas Series, a NumPy array or a
  table of one column. A linear model or a naive Bayes is scored with its own coefficients; any other model (a
  forest, an SVM, a linear `SVC`, `NuSVC` or `SVR`), and steps between the counts and it (`TruncatedSVD`,
  `SelectKBest`, a scaler), is converted by `skl2onnx` and checked. Only scikit-learn's default token pattern and
  word analyzer are taken, with `strip_accents` of `None` or `"unicode"` (`"ascii"` also deletes every character
  that is not ASCII, which Kaimei does not); before a pipeline is read, Kaimei and the vectorizer split your texts
  and Kaimei's sample texts, and one that would find different words is refused. A bag of words counts every word,
  so a text is never cut to fit: the pipeline is read for 4,096 tokens (words it finds), or for twice your longest
  text when `X` holds longer ones (up to the API's 32,768), or for `max_length=`, and Kaimei refuses a longer text
  with an error that says how long it is and how many are read.
- A pipeline that starts with a **`ColumnTransformer`** of one-hot or ordinal encoders, scalers and
  `SimpleImputer`s, each kind of column in a `Pipeline` of its own or not, or with an encoder straight into the
  model, is read exactly too, in front of any model the ONNX route reads: a column an encoder reads becomes a
  `categorical` feature with the encoder's categories (numbers and yes or no named by their text, `"3"`, `"True"`;
  infrequent categories and `drop` as scikit-learn groups and drops them). A missing value is sent as `null`: the
  model is given what the imputer fills in, the feature's `baseline`. Columns may be chosen by name, position, mask
  or slice; a part Kaimei does not write down is refused by name when columns are chosen by name, and converted
  whole by `skl2onnx` when they are chosen by position. `client.models.upload(pipeline, name=..., X=frame)`.
  The columns are computed in double precision, as scikit-learn computes them, so a time in seconds or an amount in
  cents is read as the number it is. Give rows (`X=`): the pipeline is then checked on your own values and on values
  drawn from the range of its data, and the model after the columns is read in single precision only where that
  gives the pipeline's numbers on them (`metadata.model_precision`). A pipeline fitted on an array takes
  `feature_names=` (one name for each column of that array), says which column each feature is read from in
  `metadata.table_columns`, and its reference rows are stored under the features its parts read, whatever order
  they read them in.

**What a pipeline's reading is held to.** A classifier pipeline is called exact only when, on every row of its
check, the class Kaimei reports is the class the pipeline's own `predict()` gives. A threshold wrapper
(`TunedThresholdClassifierCV`, `FixedThresholdClassifier`) has its threshold carried. Where `predict()` does not go by
`predict_proba` (a modified-Huber SGD classifier), `decision_function` is read instead. Where no reading's class is
the pipeline's, the model is drawn, not run, and its summary says on how many rows the class differs. A pipeline that
ends in a multi-label classifier is read as one, each label's probability as the model gives it.

Known limits of the pipeline readers, which each reading's summary says too:

- **A text with no words is refused.** The API needs at least one token. A text in which the vectorizer finds no
  word (no run of two or more letters or digits that is not a stop word: `"?"`, `"a b c"`, `""`) has none, and is
  refused with `Feature 'text' is empty`, where scikit-learn answers such a text from the model's intercept. Words
  the vocabulary does not hold are tokens (`<unk>`), so a text of unknown words is answered, exactly.
- **A category the encoder never saw is refused**, also when the pipeline's own encoder would let it pass
  (`handle_unknown="ignore"`, `"infrequent_if_exist"`, `use_encoded_value`): a `categorical` feature takes the
  categories the encoder learned.
- **A text longer than the reading's `max_length` is refused**, where scikit-learn has no limit. Upload with a
  larger `max_length=` (up to 32,768) to read longer ones.
- **`HashingVectorizer` pipelines are not read.** The vectorizer keeps no vocabulary to make a tokenizer of. The
  refusal says so; use a `CountVectorizer` or `TfidfVectorizer`, or register the model with an agent.
- **An SVC with `decision_function_shape="ovo"` and more than two classes is refused** behind a pipeline as on its
  own (set `decision_function_shape="ovr"`, which needs no refit).
- **An `SVC(probability=True)` behind a pipeline** is read by its probabilities. Its `predict()` goes by its margin,
  so the two can differ on some rows: where they differ on a row of the check the model is read by its
  `decision_function` instead; where they do not, the structure notes that they can (`metadata.predicted_class`)
  and the upload warns.
- **Without rows and without a scaler's statistics** (a column passed through, or only imputed or scaled by its
  largest value) nothing says where a column's values lie. The reading is then checked on made-up values, its
  summary says so, and a model that tells doubles apart is read in double precision while a tree is read in single
  precision, as scikit-learn computes each. Give rows to have it checked on yours.
- **`feature_names=` on a pipeline fitted on a table** is not used: such a pipeline reads its columns by the table's
  own names.

### Labels, boxes, masks, groups and text from text

- **Multi-label** (`task="multilabel_classification"`, or `labels=` on a model that ends in a sigmoid for each): a
  prediction lists every label's probability and which are predicted (`outputs={"threshold": 0.5}`).
- **Detection** (`task="object_detection"`): a PyTorch model's boxes and scores are joined into one output, so pass
  `output={"boxes": <its boxes>, "scores": <its scores>, "classes": <its classes, if apart>, "box_format": "cxcywh",
  "normalized": True, "score_link": "softmax"}` (names, or positions among what `forward` returns); Hugging Face
  detectors say all of it themselves. The model has to give a fixed number of boxes (DETR, YOLOS): one that cuts them
  down by non-maximum suppression inside the graph is drawn.
- **Segmentation** (`task="segmentation"`) reads per-pixel class scores, one score for each pixel, or class
  numbers. Kaimei finds the class axis itself: it follows the scores back from the output to the step that makes
  the classes (a convolution's channels, a linear layer's last axis) through whatever comes between (a
  log-softmax, a skip sum, a resize, a crop, a permute, a squeeze), and an argmax on the way makes the output
  class numbers. A convolution followed by an argmax over its classes is taken for a segmentation without `task=`.
  When the graph does not say, `labels=` does (the one axis as long as the labels), and then the shape (the one
  axis, of three, whose size differs from two equal others). An output none of these settles is refused: pass
  `mask_layout="chw"` (class scores, then height and width), `"hwc"` (height, width, then class scores) or
  `"indices"` (a class number for each pixel). A model that gives one score for each pixel (a binary segmenter) is
  cut where its own probability is one half: at 0.5 when it ends in a sigmoid, at 0 when it gives raw scores,
  whichever of `1 x 1 x H x W`, `1 x H x W`, `H x W x 1` or `H x W` it writes. Kaimei reads which off the graph, and
  off the model's own values when no step says; the structure's head carries the cut (`head.threshold`) and its
  summary says it. `threshold=` (on `models.upload`, `models.upload_structure` and `extract_structure`) says where to
  cut instead, in the model's own numbers: a pixel is the foreground at that score or more. It applies only to a mask
  of one score for each pixel (a mask of several classes or of class numbers, another task and a value that is no
  number are refused with `param="threshold"`), and it is not a classifier's `outputs={"threshold": ...}`. A model with several outputs is read from the output of floats that can hold a mask (a
  height and a width, with or without an axis of classes) and is the picture's own size, else a whole fraction or
  multiple of it, else the first such output (torchvision's `out`, not its `aux`), and from an output of whole
  numbers only when no output of floats can; `output=` reads another.
- **Clustering**: a scikit-learn KMeans or Gaussian mixture, read by its distances or its probabilities.
- **Encoder-decoders** (`task="sequence_to_sequence"`, or a Hugging Face `...ForConditionalGeneration`): T5, BART,
  Marian and their kind are exported with their decoder's input, and `client.models.generate(model, input={"text":
  ...})` writes the answer. `max_length=` sets how many tokens both the input and the answer may hold. An
  encoder-decoder kept as an ONNX file is read by saying which input is what it has written so far:
  `inputs={"decoder_input_ids": {"generated": True}}`, with `pad_id`, `start_token` and `eos_token` beside it when
  they are not 0, the first token of the sample, and none.
- **Diffusion**: `sampler=` takes a `diffusers` scheduler (or upload the pipeline itself), and `generate` then samples
  with the scheduler's own noise schedule.

### Text and language models

A model that reads text takes token ids. Pass its tokenizer and the feature becomes text:

```python
ids = tokenizer("an example", padding="max_length", max_length=128, return_tensors="pt")["input_ids"]
model = client.models.upload(
    classifier, name="Reviews", task="binary_classification",
    sample=ids, tokenizer=tokenizer,
    feature_names=["review"], labels=["negative", "positive"],
)
client.models.predict(model.id, input={"review": "the film was wonderful"})
```

A tokenizer file is never uploaded. The tokenizer is written down as JSON (its kind, vocabulary, merges and special tokens), and what Kaimei can do with it depends on its kind:

- **Run by Kaimei**: a description of your own (`{"type": "word" | "char" | "wordpiece" | "bpe", "vocab": [...]}`), a Hugging Face WordPiece tokenizer as BERT uses it, and a byte-level BPE tokenizer as GPT-2 and RoBERTa use it. A Hugging Face tokenizer is translated and then compared with the original on the sample texts of `kaimei.text.SAMPLE_TEXTS` (plain and odd spacing, punctuation, accents, other scripts, emoji, underscores, a final sigma, special and added tokens written in the text, which are matched as Hugging Face matches them, and runs of hundreds of bytes with no break); every text has to give exactly the same ids. `kaimei.text.tokenizer_spec(tokenizer, samples=[...])` does the same with texts of your own added, and its result can be passed as `tokenizer=`. The tokenizers of `bert-base-uncased`, `bert-base-cased`, `distilbert-base-uncased`, `gpt2` and `roberta-base` passed. A tokenizer with two added tokens that are the same text once normalized (`newtoken` and `NewToken` with an uncased normalizer) is sent as `pieces`: Hugging Face takes one or the other from one run to the next.
- **Vocabulary only**: any other tokenizer (SentencePiece, tiktoken, one of your own), and one that fails the comparison. The upload warns and sends `{"type": "pieces", "vocab": [...]}`. Kaimei can then name every token and turn ids back into text, so explanations read as pieces of words, but it cannot split text. Send the model token ids, or run its agent with the tokenizer (below) and send text.
- **No tokenizer**: the feature takes lists of token ids.

**The sample's length is the most tokens the model will ever be given**, because the graph's shapes are fixed. A longer text is cut to it. For a language model the prompt and what it writes share it: uploaded with a four-token sample, it stops writing after four tokens. Give a sample as long as the texts you will send, or pass `max_length=`, and the sample is filled up to it with the padding token:

```python
writer = client.models.upload(
    gpt, name="Small GPT", task="text_generation",
    sample=tokenizer("Once upon", return_tensors="pt")["input_ids"], tokenizer=tokenizer,
    feature_names=["prompt"], max_length=64,
)
client.models.generate(writer.id, input={"prompt": "Once upon"}, steps=20, temperature=0.8, top_k=40, seed=1)
client.models.tokenize(writer.id, text="Once upon")      # or ids=[...], to name tokens
```

Without `max_length` a language model's upload warns with the number of tokens it was read for. For each token it writes, the API runs the whole graph at that length, so choose a length you will use rather than the model's whole context.

### Models that stay with you: the agent

Some models are not run by Kaimei: they are too large, they need a GPU, their code is your own, or their weights may not leave your network. An agent is a small loop you start beside the model. It connects out to the Kaimei API and waits; when a prediction, an explanation or a look inside the model is asked for, the API hands it a job, the agent runs the real model, and the answer goes back. Nothing is opened on your side.

```python
from kaimei.agent import predict_with, serve

client.models.create(
    name="Churn (in our VPC)", task="binary_classification",
    features=[{"name": "tenure", "type": "numeric"}, {"name": "seats", "type": "numeric"}],
    outputs={"labels": ["stays", "leaves"]},
    inference={"type": "agent"},
)
serve(client, "churn-in-our-vpc", predict=predict_with(estimator, ["tenure", "seats"]))   # blocks; Ctrl+C stops it
```

`predict(rows)` is any function from a list of rows to `(rows,)` values or `(rows, classes)` probabilities. A row is `{feature: value}`: a number, flag or category; a list of token ids for text; a `float32` NumPy array for an image (height by width by channels, in the feature's units), a series or a tensor. While no agent is connected, requests for the model answer `502 agent_offline`; `client.models.agent_status(model)` says whether one is.

The agent can also run the model itself. It then answers trace and activations requests too, so the model's page shows live values for a structure uploaded with `weights="summary"`:

```python
model = client.models.upload(
    net, name="Encoder", task="binary_classification", sample=ids, tokenizer=tokenizer,
    weights="summary",                 # shapes and statistics: nothing Kaimei could run ("pictures" adds a sketch of each tensor)
    inference={"type": "agent"},
    save_onnx="encoder.onnx",          # the file the structure was read from, written here and not uploaded
)
serve(client, model.id, onnx="encoder.onnx", tokenizer=tokenizer)     # runs the file with ONNX Runtime
# or: serve(client, model.id, module=net, tokenizer=tokenizer)        # runs the module with forward hooks
```

- `onnx=` takes the model's ONNX file (a path, bytes or an `onnx.ModelProto`) and needs `onnxruntime`. It can read any tensor the file computes from its input, so it reports every such step of a structure that was read from that file. `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` uses a GPU. The file's inputs are fed the obvious way (the table's columns in order, the text's ids and mask, the image channels first where the file has it so); `structure=` says how every input of the file is fed, and `inputs={input name: ...}` says otherwise for the inputs it names (also beside `structure=`: every other input is still fed as the structure says; the name is the file's own, `state` and not the structure's `in.state`, and one that is not an input of the file is refused when the agent is made). With no structure on Kaimei the prediction is read where the importer reads it: the first output of floats, or what a converted scikit-learn classifier gives beside its class (behind its `ZipMap`). Its probabilities are sent as they are; a classifier without them gives scores, which are sent so that Kaimei reads from them the class the model predicts: one margin for two classes (its `decision_function`), the scores for more, or the class itself where the scores do not give it (an SVC of three classes or more decides by votes, which can tie). A file whose outputs hold only labels is refused.
- `module=` takes the `torch.nn.Module`. Forward hooks see what each module returns, so a step is reported only where a module's output is that step's. The arithmetic a `forward` writes between its modules, and the inside of an attention layer, are not reported.
- Either way the agent sends the model's own output and Kaimei reads it as the structure does: scores go through the structure's softmax or sigmoid, and a language model is read at the last token it was given. With no structure, the agent says whether the model ends in a softmax or a sigmoid (read from the module's code, the module that gave the output, or the file's last node), so its probabilities are not squashed again. A `predict` function of your own is taken to return probabilities, or the value; set its `link` attribute to `"none"` or `"softmax"` to say so for one that returns the model's own output (`raw = True`).
- `tokenizer=` lets the agent split text for a model whose tokenizer Kaimei has only the vocabulary of: a Hugging Face tokenizer, a SentencePiece processor, a tiktoken encoding, or any function from a text to its ids.
- `save_onnx=` on the upload is where the file comes from. The path has to be writable: a folder that does not exist is refused before anything is uploaded (`param="save_onnx"`). If the file still cannot be written once the model is uploaded, you get a warning that names the model, not an error.

**What an agent sends.** An agent sends what your model computes from the input it is given, and never the model. The structure that names the model's steps is kept on Kaimei, so the agent does not take its word for what a step is:

- A step is sent only where its values are computed from the row. Weights, constants, and anything worked out from those alone are left out, whatever the structure calls them. That includes what follows from the input's shape alone (the embedding of each position, cut to the length of a text) and from an input that is the same for every row (a text's segment ids, a value fixed at upload). With `onnx=` this is read off the file's graph. With `module=` the run is followed one operation at a time (PyTorch 1.13 or later).
- A step's values are laid out only in the shape its tensor has, in a true order of its axes.
- Started with `structure=` (the `Extraction` you uploaded, or its spec), an agent answers only for the steps that structure holds, and reads the prediction where that structure's head does. This is the setting to use when the weights must stay home (`weights="summary"`): what is asked of the agent is then checked against a structure Kaimei cannot change.
- A step that is kept back is named with its reason under `withheld` in the agent's answer and once in its log.

What stays visible is the model at work. Whoever can ask your model for predictions and look inside it sees what its steps compute for the inputs they choose: a layer's bias for a row of zeros, a row of an embedding table for each token sent, a column of a weight matrix for a row that is one in a single place. From enough such answers a model can be worked out. That comes with answering at all; an agent does not prevent it.

Two things to know about `module=`:

- Tensors are followed, not plain numbers: a value taken out of a tensor and put into a new one by hand (`torch.full((n,), x.max().item())`) counts as the model's own, so a step computed from that alone is kept back.
- With a PyTorch before 1.13 the run cannot be followed. The agent then keeps back what the modules' hooks can tell: a parameter or buffer given back in whole or in part, and what a module gives that was called with nothing else. An embedding of positions made on the way (`torch.arange`) is not told apart there.

For this release the same small models were registered twice, once run by the API and once by their agent, and every trace frame, activation stage and prediction compared. With `onnx=` the agent reported every step computed from the input (a transformer's two attention maps among them), and kept back the steps that hold only the model's own numbers (its position embeddings, for example). With `module=` it reported fewer steps: not the residual additions, nor anything inside attention. Everything either agent sent equalled the API's own run to float32 rounding.

From a shell, with the key in `KAIMEI_API_KEY`:

```bash
python -m kaimei.agent --model churn-in-our-vpc --predict mypackage.scoring:predict
python -m kaimei.agent --model encoder --onnx encoder.onnx --tokenizer mypackage.text:tokenizer
```

Ctrl+C or SIGTERM (how a container is stopped) stops it and closes its session.

An agent answers one job at a time, and a request waits for it up to the model's `inference.timeout_s`. An answer may hold 4,000,000 values and 128 MiB of JSON: a larger one is not sent, and the request fails saying how large it was (a language model's scores for its whole vocabulary are one value for each token of it, for each row). A request for work is held open for 20 seconds, or less when the client's `timeout` is shorter. A look inside a very large model through `onnx=` reads its steps a few hundred at a time and runs the model once for each lot, so name the steps you want (`nodes=[...]`) rather than asking for all of them. It keeps an ONNX Runtime session for each lot, so the next look is quicker, as far as `session_memory=` allows (2 GiB unless said, `--session-memory MIB` from a shell; each session holds about three copies of the weights). A look that does not fit makes again only the sessions there is no room for, and the session that gives a prediction is always kept. Give more on a machine with the memory to spare.

### Results in a notebook

A record that ends a Jupyter cell (an explanation, an insight, a model, a prediction, an analysis, or a page of them) shows as a Kaimei card: the mark, the dashboard's colours and the dashboard's numbers, formatted by the same rules. An explanation card has the prediction, the attribution bars (hover a bar for its full precision), the probabilities, the counterfactual, and the input and JSON behind collapsible sections; an image explanation adds the photo, the evidence heatmap and the focus, with its regions. Insight cards draw importance shares, drift by status, behaviour curves, the output histogram and the learned rules the way the dashboard's cards do. The cards are plain HTML and CSS, so they render in JupyterLab, classic Jupyter, VS Code and Colab, follow the host's dark theme, and link back to the record in the dashboard. Each element also carries its style inline, so a notebook committed with its outputs keeps the cards' layout on GitHub, whose viewer drops `<style>` blocks (there they show in the light theme).

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

# Outcomes come back later: attach what really happened to the logged predictions
prediction = client.predictions.create(model="credit-risk-v3", input={...}, output={"label": "approve", "probability": 0.81})
client.predictions.log_actual(prediction.id, "default")
client.predictions.log_actuals({"pred_123": "approve", "pred_124": "default"})  # or a pandas Series indexed by id

# Quality against outcomes per week: AUC, accuracy, calibration, confusion matrix (MAE / RMSE for regressions)
report = client.models.performance("credit-risk-v3", window="week", days=90)

# Has quality dropped by more than chance explains? Also part of every full analysis.
client.analyses.create(model="credit-risk-v3", type="performance")

# Study the model as a whole, then read the insights
analysis = client.analyses.create(model="credit-risk-v3", type="full")
client.analyses.wait(analysis.id)
insights = client.insights.list(model="credit-risk-v3")

# Walk every page
for model in client.models.iterate():
    print(model.slug)
```

Dashboard sign-in (session token) is also supported via `client.auth.login(...)` / `client.auth.signup(...)`. Production sign-up needs a Cloudflare Turnstile token (`turnstile_token=...`), and every sign-up needs `accept_terms=True` once the person has agreed to the [Terms of Use](https://kaimei.ai/terms). Prefer creating API keys in the dashboard.

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
| `client.organizations` | `/v1/organizations` (your organizations, joining by domain, `join-domain/confirm`) |
| `client.invitations` | `/v1/invitations` (invitations sent to you) |
| `client.projects` | `/v1/projects` (`include="summary"`, `overview(project)`) |
| `client.health()` | `/health` |

Errors raise `kaimei.KaimeiError` subclasses (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, …) and expose `.code`, `.type`, and `.status_code`. `PermissionError` remains an alias of `PermissionDeniedError`.

429 responses are retried (honouring `Retry-After`). GET requests also retry 502/503/504. Default timeout is 120 seconds.

## License

Distributed under the [`Apache 2.0 License`](https://github.com/KaimeiAI/kaimei-python-client/blob/main/LICENSE). Copyright Kaimei AI.
