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
        accept_terms: Optional[bool] = None,
    ) -> APIObject:
        """Create an account. Production sign-up requires ``turnstile_token`` when Turnstile is on.

        Pass ``accept_terms=True`` once the person has agreed to the Terms of Use (https://kaimei.ai/terms) and
        acknowledged the Privacy Policy. Kaimei API 1.2.0 refuses an account without it (``400
        terms_not_accepted``) and records it with the Terms' version; 1.1.0 records it when given.
        """
        session = self._post(
            "/v1/auth/signup",
            compact(
                {
                    "email": email,
                    "password": password,
                    "name": name,
                    "organization_name": organization_name,
                    "turnstile_token": turnstile_token,
                    "accept_terms": accept_terms,
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

    def forgot_password(self, *, email: str, turnstile_token: Optional[str] = None) -> Any:
        """Ask for a reset link. The response is the same when the email is not an account."""
        return self._post(
            "/v1/auth/forgot-password",
            compact({"email": email, "turnstile_token": turnstile_token}),
        )

    def reset_password(self, *, token: str, password: str, accept_terms: Optional[bool] = None) -> APIObject:
        """Set a new password from a reset link and sign in with it.

        When the account was never confirmed, the link makes it yours, and Kaimei API 1.2.0 asks you to agree to the
        Terms of Use as at sign-up: it answers ``400 terms_not_accepted`` until you pass ``accept_terms=True``, and
        the same link works again then. Otherwise ``accept_terms`` is ignored."""
        session = self._post(
            "/v1/auth/reset-password",
            compact({"token": token, "password": password, "accept_terms": accept_terms}),
        )
        access = getattr(session, "access_token", None)
        if access:
            self._client.use_token(access)
        return session

    def resend_verification(self, *, email: str, turnstile_token: Optional[str] = None) -> Any:
        return self._post(
            "/v1/auth/resend-verification",
            compact({"email": email, "turnstile_token": turnstile_token}),
        )

    def switch(self, organization: str) -> APIObject:
        """Work in another organization you belong to, by id or slug. ``"personal"`` is your own workspace.

        Every later call reads and writes that organization. API keys do not switch: a key works in
        the organization it was created in.
        """
        session = self._post("/v1/auth/switch", {"organization": organization})
        access = getattr(session, "access_token", None)
        if access:
            self._client.use_token(access)
        return session

    def me(self) -> APIObject:
        return self._get("/v1/auth/me")

    def update_me(self, **fields: Any) -> APIObject:
        return self._patch("/v1/auth/me", compact(fields))

    def logout(self) -> Any:
        return self._post("/v1/auth/logout", {})
