# kaimei

[![PyPI](https://img.shields.io/pypi/v/kaimei.svg)](https://pypi.org/project/kaimei/)
[![Python versions](https://img.shields.io/pypi/pyversions/kaimei.svg)](https://pypi.org/project/kaimei/)
[![License](https://img.shields.io/pypi/l/kaimei.svg)](https://github.com/KaimeiAI/kaimei-python-client/blob/main/LICENSE)

Official Python client for the [Kaimei](https://kaimei.ai) explainability API.

Kaimei (解明) — *to clarify, elucidate, reveal through investigation* — turns trained black-box models into something you can inspect: feature-level attributions, global behaviour, learned rules, robustness, and drift.

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

For a local API (the `kaimei-api` service in the [KaimeiAI/kaimei](https://github.com/KaimeiAI/kaimei) repo):

```python
client = Kaimei(api_key="km_test_...", base_url="http://localhost:8000")
```

`KAIMEI_API_KEY` and `KAIMEI_BASE_URL` are read from the environment when arguments are omitted.

## Upload a trained model from a notebook

The client reads the live estimator in your process (sklearn, Keras / TensorFlow,
PyTorch, trees / forests) and uploads the layers and learned weights as JSON.
Kaimei does not store the original ``.h5`` / ``.pt`` file.

```python
from kaimei import Kaimei

client = Kaimei(api_key="km_live_...", base_url="http://localhost:8000")

model = client.models.upload(
    clf,  # a fitted estimator
    name="Churn net",
    feature_names=["tenure", "spend", "tickets"],  # optional if the estimator knows them
    X=X_train,   # optional reference rows for explanations
    y=y_train,
)
# Open the dashboard — the architecture scene shows every layer and tensor.
```

Attach weights to a model you already registered with ``client.models.upload_structure(model.id, clf)``.

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
```

Dashboard sign-in (session token) is also supported via `client.auth.login(...)` / `client.auth.signup(...)`.

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

Errors raise `kaimei.KaimeiError` subclasses (`AuthenticationError`, `NotFoundError`, …) and expose `.code`, `.type`, and `.status_code`.

## License

Apache 2.0. Copyright Kaimei AI.
