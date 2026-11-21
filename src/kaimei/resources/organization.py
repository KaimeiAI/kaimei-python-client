from __future__ import annotations

from typing import Any, Mapping, Optional
from urllib.parse import quote

from .._objects import APIObject
from ._base import Page, Resource, compact


class Organization(Resource):
    """The organization the client is working in: your personal workspace, or a team you switched to
    with ``auth.switch``. An API key always works in the organization it was created in."""

    def retrieve(self):
        return self._get("/v1/organization")

    def update(self, *, name: Optional[str] = None, settings: Optional[Mapping[str, Any]] = None):
        return self._patch("/v1/organization", compact({"name": name, "settings": dict(settings) if settings else None}))

    def overview(self):
        return self._get("/v1/organization/overview")

    def delete(self, *, confirm: str):
        """Delete a team and everything in it. ``confirm`` is its slug. Owners only, and it cannot be undone.

        The client's session still names the deleted team afterwards: call ``auth.switch("personal")`` before
        anything else.
        """
        return self._delete(f"/v1/organization?confirm={quote(confirm, safe='')}")

    # --- members ---

    def members(self, *, limit: int = 50, offset: int = 0) -> Page:
        """Who is in the organization, with their roles."""
        return self._page("/v1/organization/members", {"limit": limit, "offset": offset})

    def update_member(self, user_id: str, *, role: str) -> APIObject:
        """Change a member's role: ``owner``, ``admin``, ``member`` or ``viewer``."""
        return self._patch(f"/v1/organization/members/{user_id}", {"role": role})

    def remove_member(self, user_id: str):
        """Remove a member. With your own user id, this leaves the organization."""
        return self._delete(f"/v1/organization/members/{user_id}")

    # --- joining by email domain ---

    def join_domain(self) -> APIObject:
        """Whether anyone with an address at this organization's email domain may join it, and as what.
        ``problem`` says why you could not turn it on right now."""
        return self._get("/v1/organization/join-domain")

    def enable_join_domain(self, *, role: str = "member") -> APIObject:
        """Let anyone with an address at your own email domain join as ``member`` or ``viewer``, once they confirm
        that address by an emailed link. When it is on already, this changes the role. Owners and admins, on Team."""
        return self._put("/v1/organization/join-domain", {"role": role})

    def disable_join_domain(self) -> APIObject:
        """Stop people joining by email domain. Links already mailed stop working; nobody is removed."""
        return self._delete("/v1/organization/join-domain")

    # --- invitations ---

    def invitations(self, *, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> Page:
        """Invitations sent from this organization. Pending ones unless ``status`` says otherwise (``all`` for every one)."""
        return self._page(
            "/v1/organization/invitations", compact({"status": status, "limit": limit, "offset": offset})
        )

    def invite(self, *, email: str, role: str = "member") -> APIObject:
        """Invite an email address as ``admin``, ``member`` or ``viewer``. The result carries ``invite_url``,
        which is shown only here, and ``email_sent``, which is false when the link was not mailed."""
        return self._post("/v1/organization/invitations", {"email": email, "role": role})

    def resend_invitation(self, invitation_id: str) -> APIObject:
        """Send an invitation again with a new link. The old link stops working."""
        return self._post(f"/v1/organization/invitations/{invitation_id}/resend")

    def revoke_invitation(self, invitation_id: str):
        return self._delete(f"/v1/organization/invitations/{invitation_id}")
