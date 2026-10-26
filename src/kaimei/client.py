"""The ``Kaimei`` client — the object you construct once and reuse."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from ._http import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, Transport
from .resources.analyses import Analyses
from .resources.api_keys import APIKeys
from .resources.auth import Auth
from .resources.explanations import Explanations
from .resources.insights import Insights
from .resources.models import Models
from .resources.organization import Organization
from .resources.predictions import Predictions
from .resources.projects import Projects

__version__ = "0.6.3"


class Kaimei:
    """Official Python client for the Kaimei explainability API.

    Typical usage matches the snippet on https://kaimei.ai ::

        from kaimei import Kaimei

        client = Kaimei(api_key="km_live_...")
        explanation = client.explanations.create(
            model="credit-risk-v3",
            input={"income": 72000, "credit_score": 681, "utilization": 0.41},
        )
        print(explanation.output)
        print(explanation.top_features)

    Credentials are read from ``KAIMEI_API_KEY`` (and optionally
    ``KAIMEI_BASE_URL``) when not passed explicitly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        default_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("KAIMEI_API_KEY")
        url = (base_url or os.environ.get("KAIMEI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._transport = Transport(
            api_key=key,
            base_url=url,
            timeout=timeout,
            max_retries=max_retries,
            user_agent=f"kaimei-python/{__version__}",
            extra_headers=default_headers,
        )
        self.explanations = Explanations(self)
        self.models = Models(self)
        self.projects = Projects(self)
        self.predictions = Predictions(self)
        self.analyses = Analyses(self)
        self.insights = Insights(self)
        self.auth = Auth(self)
        self.api_keys = APIKeys(self)
        self.organization = Organization(self)

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    def use_token(self, token: str) -> None:
        """Switch the Authorization header (used after ``auth.login`` / ``auth.signup``)."""
        self._transport.api_key = token
        self._transport._client.headers["Authorization"] = f"Bearer {token}"

    def health(self) -> Any:
        """``GET /health`` — liveness of the API process."""
        from ._objects import wrap

        return wrap(self._request("GET", "/health"))

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "Kaimei":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        content: Optional[bytes] = None,
        raw: bool = False,
    ) -> Any:
        return self._transport.request(method, path, params=params, json_body=json_body, content=content, raw=raw)
