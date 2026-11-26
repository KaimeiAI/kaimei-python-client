from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from .._objects import APIObject
from ._base import Page, Resource, compact


class Organizations(Resource):
    """The organizations the signed-in account belongs to. These calls need a session
    (``auth.login``), not an API key."""

    def list(self, *, limit: int = 50, offset: int = 0) -> Page:
        """Your personal workspace first, then your organizations by name, each with your ``role`` in it."""
        return self._page("/v1/organizations", {"limit": limit, "offset": offset})

    def create(self, *, name: str, slug: Optional[str] = None) -> APIObject:
        """Create an organization you own. Call ``auth.switch`` to start working in it."""
        return self._post("/v1/organizations", compact({"name": name, "slug": slug}))

    def joinable(self, *, limit: int = 50, offset: int = 0) -> Page:
        """Teams you may join because your address is at their email domain, and that you are not in yet."""
        return self._page("/v1/organizations/joinable", {"limit": limit, "offset": offset})

    def join(self, organization_id: str) -> APIObject:
        """Ask to join one of ``joinable()``. A confirmation link is emailed to your address; the link is never
        returned. Open it, or pass its token to ``confirm_join``, signed in as the same account."""
        return self._post(f"/v1/organizations/{quote(organization_id, safe='')}/join")

    def confirm_join(self, token: str) -> APIObject:
        """Join with the token from the emailed link. Returns the organization; ``auth.switch`` to work in it."""
        return self._post("/v1/organizations/join/confirm", {"token": token})

    def confirm_join_domain(self, token: str) -> APIObject:
        """Turn joining by email domain on, with the token from the link ``organization.enable_join_domain``
        emailed you. Signed in as the account that asked, and still an owner or admin there; the client can be
        working in any organization. Returns that organization's setting. Needs Kaimei API 1.2.0 or newer."""
        return self._post("/v1/organizations/join-domain/confirm", {"token": token})


class Invitations(Resource):
    """Invitations sent to the signed-in account. Sending them is ``organization.invite``."""

    def list(self, *, limit: int = 50, offset: int = 0) -> Page:
        """Invitations waiting for an answer."""
        return self._page("/v1/invitations", {"limit": limit, "offset": offset})

    def preview(self, token: str) -> APIObject:
        """What an invitation link offers. Needs no credentials."""
        return self._post("/v1/invitations/preview", {"token": token})

    def accept(self, *, token: Optional[str] = None, invitation_id: Optional[str] = None) -> APIObject:
        """Join the organization, by the link's token or by the invitation's id. Returns the organization."""
        return self._post("/v1/invitations/accept", _which(token, invitation_id))

    def decline(self, *, token: Optional[str] = None, invitation_id: Optional[str] = None) -> None:
        self._post("/v1/invitations/decline", _which(token, invitation_id))


def _which(token: Optional[str], invitation_id: Optional[str]) -> dict:
    if bool(token) == bool(invitation_id):
        raise ValueError("Pass either token or invitation_id.")
    return {"token": token} if token else {"invitation_id": invitation_id}
