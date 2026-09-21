from __future__ import annotations

from typing import Iterable, Optional

from ._base import Resource, compact


class APIKeys(Resource):
    def list(self):
        return self._get("/v1/api-keys")

    def create(
        self,
        *,
        name: str,
        environment: str = "live",
        scopes: Optional[Iterable[str]] = None,
        expires_in_days: Optional[int] = None,
    ):
        return self._post(
            "/v1/api-keys",
            compact(
                {
                    "name": name,
                    "environment": environment,
                    "scopes": list(scopes) if scopes is not None else None,
                    "expires_in_days": expires_in_days,
                }
            ),
        )

    def revoke(self, key_id: str):
        return self._delete(f"/v1/api-keys/{key_id}")
