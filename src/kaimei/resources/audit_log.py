from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from ._base import Page, Resource, compact

When = Union[str, datetime, date, None]


def _instant(value: When) -> Optional[str]:
    """A moment as the API reads it. A day means its first instant, and a time without a zone is UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return moment.isoformat()
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()


class AuditLog(Resource):
    """``/v1/audit-log`` — who did what in the organization, and when.

    Signing in, creating and revoking keys, adding and removing people, changing models, approving reason
    tables, recording what was decided, placing and releasing holds, and every deletion, by a person or by the
    retention schedule. Entries are written with the change they describe and cannot be edited or deleted.
    Owners, admins and reviewers read the log; an API key reads it when the person who created it may.
    """

    def _params(
        self,
        action: Optional[str],
        actor: Optional[str],
        target_type: Optional[str],
        target_id: Optional[str],
        model: Optional[str],
        since: When,
        until: When,
    ) -> Dict[str, Any]:
        return {
            "action": action,
            "actor": actor,
            "target_type": target_type,
            "target_id": target_id,
            "model": model,
            "since": _instant(since),
            "until": _instant(until),
        }

    def list(
        self,
        *,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        model: Optional[str] = None,
        since: When = None,
        until: When = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        """Entries, newest first.

        ``action`` is one action such as ``"model.updated"``, or a group such as ``"model"``, ``"decision"``,
        ``"hold"``, ``"retention"``, ``"api_key"``, ``"member"`` or ``"auth"``. ``actor`` is part of a name or an
        address. ``target_type`` and ``target_id`` find what was done to one thing:
        ``target_type="decision", target_id="APP-2026-00417"``. ``since`` and ``until`` take a time, a date or text.
        """
        params = self._params(action, actor, target_type, target_id, model, since, until)
        return self._page("/v1/audit-log", {**params, "limit": limit, "offset": offset})

    def iterate(self, **params: Any) -> Iterator[Any]:
        """Every matching entry, newest first, a page at a time. Takes what :meth:`list` takes."""
        params.pop("offset", None)
        for key in ("since", "until"):
            if key in params:
                params[key] = _instant(params[key])
        return self._iter("/v1/audit-log", params)

    def csv(
        self,
        *,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        model: Optional[str] = None,
        since: When = None,
        until: When = None,
    ) -> bytes:
        """The matching entries as a CSV file, newest first, at most 10,000 of them. Opens in a spreadsheet."""
        params = compact(
            {
                **self._params(action, actor, target_type, target_id, model, since, until),
                "format": "csv",
            }
        )
        return bytes(self._client._request("GET", "/v1/audit-log", params=params, raw=True))

    def save_csv(self, path: Union[str, Path] = "audit-log.csv", **filters: Any) -> Path:
        """Save the matching entries as a CSV file and return where it was written."""
        target = Path(path)
        if target.is_dir():
            target = target / "audit-log.csv"
        target.write_bytes(self.csv(**filters))
        return target
