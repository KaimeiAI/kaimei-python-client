from __future__ import annotations

from typing import Any, Dict, Iterator, Optional
from urllib.parse import quote

from .._objects import APIObject
from ._base import Page, Resource, compact


class Retention(Resource):
    """``/v1/retention`` — how long logged decisions are kept, and the holds that keep them longer.

    A schedule deletes a model's logged decisions, with their explanations and what was recorded as decided,
    once they are older than the period set for the model or, failing that, for the organization. With no
    schedule, records are kept until someone deletes them. A hold keeps one decision, or every record of a
    model, whatever the schedule says, and stops anyone deleting it by hand, until it is released. Every
    deletion is entered in the audit log. Models, analyses, insights, reason tables and the audit log are never
    deleted by a schedule.
    """

    def retrieve(self) -> APIObject:
        """The schedule, model by model: each model's period, how many decisions it holds and how old the
        oldest is, what the schedule would delete now (``due``), and what is under a hold."""
        return self._get("/v1/retention")

    def set(self, keep_days: Optional[int]) -> APIObject:
        """Set the organization's period: delete records ``keep_days`` after they are logged.

        ``None`` goes back to keeping records until someone deletes them. Nothing is deleted by this call: the
        schedule is applied a few times a day, and :meth:`run` applies it now. An admin or owner sets it, signed
        in: an API key is refused. Sign in with ``client.auth.login``, or set it in the dashboard.
        """
        return self._put("/v1/retention", {"keep_days": keep_days})

    def set_for_model(self, model: str, keep_days: Optional[int]) -> APIObject:
        """Give a model its own period. ``None`` keeps this model's records until someone deletes them, whatever
        the organization's period."""
        return self._put(
            f"/v1/retention/models/{quote(str(model), safe='')}", {"keep_days": keep_days}
        )

    def clear_for_model(self, model: str) -> APIObject:
        """Have a model follow the organization's period again."""
        return self._delete(f"/v1/retention/models/{quote(str(model), safe='')}")

    def run(self, *, dry_run: bool = False) -> APIObject:
        """Apply the schedule now and say how much was deleted, model by model.

        ``dry_run=True`` deletes nothing and says what would be. Records under a hold are kept.
        """
        path = "/v1/retention/run" + ("?dry_run=true" if dry_run else "")
        return self._post(path)

    # --- holds ---

    def holds(
        self,
        *,
        active: Optional[bool] = None,
        model: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        """Holds, those in force first. ``active=True`` lists only those in force."""
        return self._page(
            "/v1/retention/holds",
            {
                "active": active,
                "model": model,
                "decision": decision,
                "limit": limit,
                "offset": offset,
            },
        )

    def iterate_holds(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/retention/holds", params)

    def place_hold(
        self,
        *,
        reason: str,
        decision: Optional[str] = None,
        model: Optional[str] = None,
        matter: Optional[str] = None,
    ) -> APIObject:
        """Hold one decision's record (``decision``), or every record a model has logged (``model``).

        Name exactly one of the two. ``reason`` says why; ``matter`` is your own reference, such as a case
        number. A held record cannot be deleted by the schedule or by hand, and a model with a hold on its
        records cannot be deleted. Members, reviewers, admins and owners may place a hold.
        """
        if bool(decision) == bool(model):
            raise ValueError(
                "Name what to hold: decision=..., or model=... for every record it has."
            )
        body: Dict[str, Any] = {
            "decision": decision,
            "model": model,
            "reason": reason,
            "matter": matter,
        }
        return self._post("/v1/retention/holds", compact(body))

    def release_hold(self, hold_id: str) -> APIObject:
        """Release a hold. It stops applying and is kept, with who released it and when.

        A reviewer, admin or owner releases a hold, signed in: an API key is refused. Sign in with
        ``client.auth.login``, or release it in the dashboard.
        """
        return self._post(f"/v1/retention/holds/{quote(str(hold_id), safe='')}/release")
