from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Union

from .._objects import APIObject
from ._base import Page, Resource, compact

#: An input's approved wording: the words themselves, or ``{"reason": words, "code": your code for it}``.
Wording = Union[str, Mapping[str, Any]]


class ReasonTables(Resource):
    """``/v1/reason-tables`` — the approved wording for why a model decided against someone.

    A reason table says, for each input of a model, what a person is told when that input pushed the decision
    against them, and which inputs must never be given as a reason. A draft changes nothing; once a reviewer,
    admin or owner approves it in the dashboard it is in force, and each decision record draws its draft
    reasons with the version that was in force when the decision was made (``record.reasons``).
    """

    def list(
        self,
        *,
        model: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        """Versions of reason tables, most recently changed first. ``status`` is ``draft``, ``approved`` or
        ``retired``; ``status="approved"`` gives the versions in force, one for each model that has one."""
        return self._page(
            "/v1/reason-tables",
            {"model": model, "status": status, "limit": limit, "offset": offset},
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/reason-tables", params)

    def retrieve(self, table_id: str) -> APIObject:
        return self._get(f"/v1/reason-tables/{table_id}")

    def in_force(self, model: str) -> Optional[APIObject]:
        """The approved version that applies to the model's decisions now, or ``None`` when it has none."""
        found = self.list(model=model, status="approved", limit=1).items
        return found[0] if found else None

    def draft(self, model: str) -> Optional[APIObject]:
        """The model's draft, or ``None`` when nobody is working on one."""
        found = self.list(model=model, status="draft", limit=1).items
        return found[0] if found else None

    def save_draft(
        self,
        model: str,
        *,
        entries: Mapping[str, Wording],
        adverse_outcome: Optional[str] = None,
        adverse_direction: Optional[str] = None,
        never_state: Union[Mapping[str, str], Iterable[str], None] = None,
        max_reasons: int = 4,
        note: Optional[str] = None,
    ) -> APIObject:
        """Create the model's draft, or replace what it says. The version in force is not touched.

        ``entries`` maps an input's name to its approved wording. ``adverse_outcome`` names the outcome that
        goes against the person, such as ``"decline"``; a model that predicts a number takes
        ``adverse_direction`` instead, ``"higher"`` or ``"lower"``. ``never_state`` lists the inputs that must
        never be given as a reason (a mapping adds a word on why). An input in neither is undecided, and is held
        for review on a decision's record whenever it pushes toward the adverse outcome.
        """
        if (adverse_outcome is None) == (adverse_direction is None):
            raise ValueError(
                "Give adverse_outcome for a model that decides (such as 'decline'), or adverse_direction "
                "('higher' or 'lower') for one that predicts a number."
            )
        worded: Dict[str, Dict[str, Any]] = {}
        for name, entry in entries.items():
            worded[str(name)] = {"reason": entry} if isinstance(entry, str) else dict(entry)
        if never_state is None:
            never: Dict[str, str] = {}
        elif isinstance(never_state, Mapping):
            never = {str(name): str(why or "") for name, why in never_state.items()}
        elif isinstance(never_state, str):
            never = {never_state: ""}
        else:
            never = {str(name): "" for name in never_state}
        adverse: Dict[str, Any] = (
            {"outcome": adverse_outcome}
            if adverse_outcome is not None
            else {"direction": adverse_direction}
        )
        body = {
            "model": model,
            "adverse": adverse,
            "max_reasons": max_reasons,
            "entries": worded,
            "never_state": never,
            "note": note,
        }
        return self._put("/v1/reason-tables/draft", compact(body))

    def approve(self, table_id: str) -> APIObject:
        """Put a draft in force. It takes a signed-in reviewer, admin or owner: an API key is refused, since the
        table records the person who approved it. Sign in with ``client.auth.login`` or approve in the dashboard."""
        return self._post(f"/v1/reason-tables/{table_id}/approve")

    def discard(self, table_id: str) -> APIObject:
        """Delete a draft. An approved or retired version is part of the record and is kept."""
        return self._delete(f"/v1/reason-tables/{table_id}")
