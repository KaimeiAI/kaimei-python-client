from __future__ import annotations

from typing import Any, Optional

from .._objects import APIObject
from ._base import Resource, compact


class Auth(Resource):
    def providers(self) -> APIObject:
        return self._get("/v1/auth/providers")

    def signup(
        self,
        *,
        email: str,
        password: str,
        name: Optional[str] = None,
        organization_name: Optional[str] = None,
        turnstile_token: Optional[str] = None,
    ) -> APIObject:
        """Create an account. Production sign-up requires ``turnstile_token`` when Turnstile is on."""
        session = self._post(
            "/v1/auth/signup",
            compact(
                {
                    "email": email,
                    "password": password,
                    "name": name,
                    "organization_name": organization_name,
                    "turnstile_token": turnstile_token,
                }
            ),
        )
        token = getattr(session, "access_token", None)
        if token:
            self._client.use_token(token)
        return session

    def login(self, *, email: str, password: str) -> APIObject:
        session = self._post("/v1/auth/login", {"email": email, "password": password})
        token = getattr(session, "access_token", None)
        if token:
            self._client.use_token(token)
        return session

    def verify_email(self, *, token: str) -> APIObject:
        session = self._post("/v1/auth/verify-email", {"token": token})
        access = getattr(session, "access_token", None)
        if access:
            self._client.use_token(access)
        return session

    def resend_verification(self, *, email: str, turnstile_token: Optional[str] = None) -> Any:
        return self._post(
            "/v1/auth/resend-verification",
            compact({"email": email, "turnstile_token": turnstile_token}),
        )

    def me(self) -> APIObject:
        return self._get("/v1/auth/me")

    def update_me(self, **fields: Any) -> APIObject:
        return self._patch("/v1/auth/me", compact(fields))

    def logout(self) -> Any:
        return self._post("/v1/auth/logout", {})
