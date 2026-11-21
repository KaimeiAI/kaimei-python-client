"""Organizations, members, invitations, and moving work between them: the calls go where the API expects."""

from __future__ import annotations

import json
from typing import Any, List, Tuple

import httpx
import pytest

from kaimei import Kaimei

ORG = {"object": "organization", "id": "org_team", "name": "Acme", "slug": "acme", "plan": "free", "kind": "team"}
SESSION = {
    "access_token": "session-in-acme",
    "token_type": "bearer",
    "expires_in": 60,
    "user": {"object": "user", "id": "usr_1", "role": "admin"},
    "organization": {**ORG, "role": "admin"},
}
MOVED = {"object": "transfer", "organization": ORG, "project": None, "models": [{"object": "model", "id": "mdl_1"}]}


def _page(items: list) -> dict:
    return {"object": "list", "data": items, "total": len(items), "limit": 50, "offset": 0, "has_more": False}


def _client(seen: List[Tuple[str, str, Any, str]]) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        seen.append((request.method, path, body, request.headers.get("Authorization", "")))
        if path == "/v1/auth/switch":
            return httpx.Response(200, json=SESSION)
        if path in {"/v1/organizations", "/v1/organization/members", "/v1/organization/invitations", "/v1/invitations"}:
            if request.method == "GET":
                return httpx.Response(200, json=_page([{"object": "row", "id": "row_1", "role": "member"}]))
            return httpx.Response(201, json={**ORG, "invite_url": "https://dashboard.kaimei.ai/invite?token=t", "email_sent": False})
        if path == "/v1/invitations/decline":
            return httpx.Response(204)
        if path.endswith(("/transfer", "/fork")):
            return httpx.Response(200, json={**MOVED, "object": path.rsplit("/", 1)[1], "credentials_removed": []})
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "x", "object": "deleted", "deleted": True})
        return httpx.Response(200, json=ORG)

    c = Kaimei(api_key="first-session", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer first-session", "Accept": "application/json"},
    )
    return c


def test_switching_makes_later_calls_use_the_new_session() -> None:
    seen: list = []
    client = _client(seen)
    session = client.auth.switch("acme")
    assert seen[-1] == ("POST", "/v1/auth/switch", {"organization": "acme"}, "Bearer first-session")
    assert session.organization.role == "admin"
    client.organization.retrieve()
    assert seen[-1][3] == "Bearer session-in-acme"


def test_listing_and_creating_organizations() -> None:
    seen: list = []
    client = _client(seen)
    assert client.organizations.list().data[0].id == "row_1"
    assert seen[-1][:2] == ("GET", "/v1/organizations")
    client.organizations.create(name="Acme")
    assert seen[-1][:3] == ("POST", "/v1/organizations", {"name": "Acme"})
    client.organizations.create(name="Acme", slug="acme-risk")
    assert seen[-1][2] == {"name": "Acme", "slug": "acme-risk"}


def test_members_and_invitations() -> None:
    seen: list = []
    client = _client(seen)
    assert client.organization.members().data[0].role == "member"
    client.organization.update_member("usr_2", role="viewer")
    assert seen[-1][:3] == ("PATCH", "/v1/organization/members/usr_2", {"role": "viewer"})
    client.organization.remove_member("usr_2")
    assert seen[-1][:2] == ("DELETE", "/v1/organization/members/usr_2")

    sent = client.organization.invite(email="bob@example.com", role="admin")
    assert seen[-1][:3] == ("POST", "/v1/organization/invitations", {"email": "bob@example.com", "role": "admin"})
    assert sent.invite_url.endswith("token=t")
    assert sent.email_sent is False
    client.organization.invitations(status="all")
    assert seen[-1][:2] == ("GET", "/v1/organization/invitations")
    client.organization.resend_invitation("inv_1")
    assert seen[-1][:2] == ("POST", "/v1/organization/invitations/inv_1/resend")
    client.organization.revoke_invitation("inv_1")
    assert seen[-1][:2] == ("DELETE", "/v1/organization/invitations/inv_1")
    client.organization.delete(confirm="acme")
    assert seen[-1][:2] == ("DELETE", "/v1/organization")


def test_joining_by_email_domain() -> None:
    seen: list = []
    client = _client(seen)
    client.organization.join_domain()
    assert seen[-1][:2] == ("GET", "/v1/organization/join-domain")
    client.organization.enable_join_domain(role="viewer")
    assert seen[-1][:3] == ("PUT", "/v1/organization/join-domain", {"role": "viewer"})
    client.organization.disable_join_domain()
    assert seen[-1][:2] == ("DELETE", "/v1/organization/join-domain")
    client.organizations.joinable()
    assert seen[-1][:2] == ("GET", "/v1/organizations/joinable")
    client.organizations.join("org_team")
    assert seen[-1][:3] == ("POST", "/v1/organizations/org_team/join", None)
    client.organizations.confirm_join("tok")
    assert seen[-1][:3] == ("POST", "/v1/organizations/join/confirm", {"token": "tok"})


def test_answering_an_invitation_takes_a_token_or_an_id_and_not_both() -> None:
    seen: list = []
    client = _client(seen)
    client.invitations.list()
    assert seen[-1][:2] == ("GET", "/v1/invitations")
    client.invitations.preview("tok")
    assert seen[-1][:3] == ("POST", "/v1/invitations/preview", {"token": "tok"})
    client.invitations.accept(token="tok")
    assert seen[-1][:3] == ("POST", "/v1/invitations/accept", {"token": "tok"})
    assert client.invitations.decline(invitation_id="inv_1") is None
    assert seen[-1][:3] == ("POST", "/v1/invitations/decline", {"invitation_id": "inv_1"})
    with pytest.raises(ValueError):
        client.invitations.accept()
    with pytest.raises(ValueError):
        client.invitations.accept(token="tok", invitation_id="inv_1")


def test_transferring_and_forking_models_and_projects() -> None:
    seen: list = []
    client = _client(seen)
    moved = client.models.transfer("credit-risk", organization="acme")
    assert seen[-1][:3] == (
        "POST",
        "/v1/models/credit-risk/transfer",
        {"organization": "acme", "include_versions": True},
    )
    assert moved.object == "transfer"
    assert moved.models[0].id == "mdl_1"
    client.models.transfer("credit-risk", organization="acme", project="lending", include_versions=False)
    assert seen[-1][2] == {"organization": "acme", "project": "lending", "include_versions": False}

    forked = client.models.fork("credit-risk")
    assert seen[-1][:3] == ("POST", "/v1/models/credit-risk/fork", {"include_versions": True})
    assert forked.object == "fork"
    assert forked.credentials_removed == []
    client.models.fork("credit-risk", organization="acme", name="My copy")
    assert seen[-1][2] == {"organization": "acme", "name": "My copy", "include_versions": True}

    client.projects.transfer("lending", organization="acme")
    assert seen[-1][:3] == ("POST", "/v1/projects/lending/transfer", {"organization": "acme"})
    client.projects.fork("lending")
    assert seen[-1][:3] == ("POST", "/v1/projects/lending/fork", {})
    client.projects.fork("lending", organization="personal", name="Lending copy")
    assert seen[-1][2] == {"organization": "personal", "name": "Lending copy"}
