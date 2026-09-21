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

from .client import Kaimei, __version__
from .errors import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    KaimeiError,
    NotFoundError,
    PermissionError,
    RateLimitError,
)
from ._objects import APIObject, Page

__all__ = [
    "Kaimei",
    "KaimeiError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "RateLimitError",
    "APIConnectionError",
    "APIStatusError",
    "APIObject",
    "Page",
    "__version__",
]
