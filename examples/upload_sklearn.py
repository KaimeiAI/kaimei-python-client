"""Upload a fitted sklearn model, then explain one row.

Requires a Kaimei API key (dashboard → API keys) and NumPy + scikit-learn::

    pip install "kaimei[models]" scikit-learn
    set KAIMEI_API_KEY=km_live_...
    python examples/upload_sklearn.py
"""

from __future__ import annotations

import os

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kaimei import Kaimei


def main() -> None:
    bundle = load_breast_cancer()
    names = list(bundle.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.data, bundle.target, test_size=0.25, random_state=0, stratify=bundle.target
    )
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    clf.fit(X_train, y_train)
    print(f"hold-out accuracy: {clf.score(X_test, y_test):.3f}")

    client = Kaimei(
        api_key=os.environ.get("KAIMEI_API_KEY"),
        base_url=os.environ.get("KAIMEI_BASE_URL") or "https://api.kaimei.ai",
    )
    model = client.models.upload(
        clf,
        name="Breast cancer logistic",
        feature_names=names,
        X=X_train[:400],
        y=y_train[:400],
    )
    print(f"registered {model.id}  slug={model.slug}")
    print("open the dashboard — the architecture scene shows the scaler and linear layer")

    row = {name: float(value) for name, value in zip(names, X_test[0])}
    explanation = client.explanations.create(model=model.slug or model.id, input=row, method="shapley")
    print(explanation.output)
    print(explanation.top_features[:5])


if __name__ == "__main__":
    if not os.environ.get("KAIMEI_API_KEY"):
        raise SystemExit("Set KAIMEI_API_KEY (and optionally KAIMEI_BASE_URL) before running.")
    main()
