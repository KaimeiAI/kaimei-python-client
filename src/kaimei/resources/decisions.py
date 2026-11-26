from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union
from urllib.parse import quote

from .._objects import APIObject
from ._base import Page, Resource, compact


def _path(decision_id: str) -> str:
    """A decision id as a path: every character escaped but the slashes an id may hold."""
    wanted = str(decision_id).strip()
    if not wanted:
        raise ValueError("A decision id is needed: the id the prediction was logged under.")
    return "/v1/decisions/" + "/".join(quote(part, safe="") for part in wanted.split("/"))


def record_filename(decision_id: str) -> str:
    """The name Kaimei gives a decision record saved as a PDF."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(decision_id)).strip("-.") or "decision"
    return f"decision-record-{safe[:80]}.pdf"


class Decisions(Resource):
    """``/v1/decisions`` — one logged decision and how the model reached it, for the people who answer for it.

    A decision is a logged prediction, known by the id it was logged under: yours (``prediction_id``), or the one
    Kaimei generated. Its record is written to be read and filed: the outcome, each input's effect in a sentence,
    what would have changed the outcome, how it was computed, and a fingerprint of the stored facts.
    """

    def retrieve(
        self, decision_id: str, *, explanation: Optional[str] = None, reason_table: Union[str, int, None] = None
    ) -> APIObject:
        """The record of one decision.

        It shows the newest explanation stored for the decision unless ``explanation`` names another. A decision
        that is logged and not explained has a record too, with ``status == "not_explained"``. Nothing is
        recomputed, so the same record has the same ``fingerprint`` every time.

        When the model has an approved reason table, ``record.reasons`` holds the draft reasons for a decision
        that went against the person, drawn with the version in force when it was made. ``reason_table="draft"``
        draws them with the model's draft instead, to try wording before it is approved; a number picks a version.
        """
        params = {"explanation": explanation, "reason_table": None if reason_table is None else str(reason_table)}
        return self._get(_path(decision_id), compact(params))

    def list(
        self,
        *,
        model: Optional[str] = None,
        q: Optional[str] = None,
        explained: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Page:
        """Logged decisions, newest first, each with whether an explanation is stored for it.

        ``q`` finds decisions whose id contains the text, whatever its case. ``explained=False`` lists the
        decisions that still have no explanation.
        """
        return self._page(
            "/v1/decisions",
            {"model": model, "q": q, "explained": explained, "limit": limit, "offset": offset},
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/decisions", params)

    def explain(
        self,
        decision_id: str,
        *,
        method: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> APIObject:
        """Explain a decision that is already logged, as it was logged, and return its record.

        With nothing chosen, Kaimei explains it the way a record that may have to be defended should be:
        KernelSHAP where the model allows it, seed 0, every input reported, and a search for what would have
        changed the outcome. ``options`` are an explanation's options, such as ``{"frozen_features": ["age"]}``
        to hold inputs fixed in that search. A reviewer may call this.
        """
        body: Dict[str, Any] = {"method": method, "options": dict(options) if options else None}
        sent = compact(body)
        return self._post(_path(decision_id) + "/explanation", sent or None)

    def pdf(self, decision_id: str, *, explanation: Optional[str] = None, paper: str = "letter") -> bytes:
        """The record as a PDF document, as bytes. ``paper`` is ``"letter"`` or ``"a4"``.

        The document states the same facts and sentences as :meth:`retrieve`, with the fingerprint on every page.
        """
        if paper not in ("letter", "a4"):
            raise ValueError("paper is 'letter' or 'a4'.")
        params = compact({"explanation": explanation, "format": "pdf", "paper": paper})
        data = self._client._request("GET", _path(decision_id), params=params, raw=True)
        return bytes(data)

    def save_pdf(
        self,
        decision_id: str,
        path: Union[str, Path, None] = None,
        *,
        explanation: Optional[str] = None,
        paper: str = "letter",
    ) -> Path:
        """Save the record as a PDF and return where it was written.

        ``path`` is a file, or a folder to put ``decision-record-<id>.pdf`` in. Left out, the file goes in the
        current folder under that name.
        """
        target = Path(path) if path is not None else Path(record_filename(decision_id))
        if target.is_dir():
            target = target / record_filename(decision_id)
        target.write_bytes(self.pdf(decision_id, explanation=explanation, paper=paper))
        return target
