"""Kaimei (解明) — official Python client for the explainability API.

Install from PyPI::

    pip install kaimei

Then::

    from kaimei import Kaimei

    client = Kaimei(api_key="km_live_...")
    explanation = client.explanations.create(
        model="credit-risk-v3",
        input={"income": 72000, "credit_score": 681, "utilization": 0.41},
    )
"""

from ._objects import APIObject, Page
from .client import Kaimei, __version__
from .errors import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    KaimeiError,
    NotFoundError,
    PermissionDeniedError,
    PermissionError,
    RateLimitError,
)
from .structure import StructureError, estimator_to_spec

__all__ = [
    "Kaimei",
    "KaimeiError",
    "AuthenticationError",
    "PermissionDeniedError",
    "PermissionError",
    "NotFoundError",
    "RateLimitError",
    "APIConnectionError",
    "APIStatusError",
    "APIObject",
    "Page",
    "StructureError",
    "estimator_to_spec",
    "__version__",
]
