from __future__ import annotations

from typing import Optional

from ._base import Resource, compact


class Auth(Resource):
    def providers(self):
        return self._get("/v1/auth/providers")

    def signup(
        self,
        *,
        email: str,
        password: str,
        name: Optional[str] = None,
        organization_name: Optional[str] = None,
    ):
        session = self._post(
            "/v1/auth/signup",
            compact(
                {
                    "email": email,
                    "password": password,
                    "name": name,
                    "organization_name": organization_name,
                }
            ),
        )
        token = getattr(session, "access_token", None)
        if token:
            self._client.use_token(token)
        return session

    def login(self, *, email: str, password: str):
        session = self._post("/v1/auth/login", {"email": email, "password": password})
        token = getattr(session, "access_token", None)
        if token:
            self._client.use_token(token)
        return session

    def me(self):
        return self._get("/v1/auth/me")

    def update_me(self, **fields):
        return self._patch("/v1/auth/me", compact(fields))

    def logout(self):
        return self._post("/v1/auth/logout", {})
