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

The client reads the live estimator in your process and uploads layers and learned weights as JSON. Kaimei does **not** store the original `.h5` / `.pt` / pickle file, and it does not host a binary artifact.

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

A runnable walkthrough lives in [`examples/upload_sklearn.py`](examples/upload_sklearn.py) and [`examples/upload_sklearn.ipynb`](examples/upload_sklearn.ipynb).

### Supported estimators

| Estimator | What is uploaded | Executable for explain / predict |
| --- | --- | --- |
| sklearn linear (`coef_` / `intercept_`) | Exact weights | Yes |
| sklearn `MLPClassifier` / `MLPRegressor` | Exact layers | Yes |
| sklearn tree | Full node list | Yes |
| sklearn forest / extra trees | Trees if ≤30 estimators and ≤4,000 nodes; otherwise importances | Yes only when trees are inlined |
| sklearn `Pipeline` + `StandardScaler` or `(0, 1)` `MinMaxScaler` | Scaler folded or inserted, then the estimator | Same as the final step |
| Keras / TensorFlow `Dense`, `Conv2D`, pooling, flatten, dropout, LayerNorm | Layer weights | Yes, unless a layer is unrecognized (`approximate`) |
| `torch.nn` `Linear`, common activations, Dropout, LayerNorm, Embedding | Forward-hook order | Yes, unless a layer is unrecognized (`approximate`) |
| XGBoost / LightGBM / CatBoost (sklearn API) | `feature_importances_` summary | No — pass `inference` remote or attributions |
| ONNX, Hugging Face, LLMs | Framework label only | No |

Unknown Keras / PyTorch layers are passed through and marked `approximate`. Those models visualize but cannot be queried unless you add a remote URL.

Large payloads (weights inlined as JSON) warn above ~2 MB and may fail at the proxy or database above ~16 MB.

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
| `client.projects` | `/v1/projects` |
| `client.health()` | `/health` |

Errors raise `kaimei.KaimeiError` subclasses (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, …) and expose `.code`, `.type`, and `.status_code`. `PermissionError` remains an alias of `PermissionDeniedError`.

429 responses are retried (honouring `Retry-After`). GET requests also retry 502/503/504. Default timeout is 120 seconds.

## License

Apache 2.0. Copyright Kaimei AI.
