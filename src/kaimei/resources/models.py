from __future__ import annotations

import dataclasses
import numbers
import os
import re
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .._http import encode_json
from .._objects import APIObject
from ..errors import APIStatusError, KaimeiError
from ..structure import (
    EXTERNALIZE_ABOVE,
    Extraction,
    StructureError,
    _input_width,
    _names_it_was_fitted_on,
    cap_reference_rows,
    detect_framework,
    encode_tensor,
    extract_structure,
    features_from_spec,
    finalize_spec,
    image_shape_of,
    label_names,
    labels_from_target,
    rows_from_inputs,
    spec_is_executable,
    tensor_dtype,
    warn_if_large_spec,
)
from ._base import Page, Resource, compact


def _plainly_named(X: Any, estimator: Any) -> bool:
    """Whether ``X`` names the model: a table, and an estimator with no names of its own for its order to match."""
    columns = getattr(X, "columns", None)
    if columns is None:
        return False
    fitted = _names_it_was_fitted_on(estimator)
    if fitted is None:
        return True
    # LightGBM keeps a column's name with underscores for its spaces, and the table has what it's really called.
    written = [str(column) for column in columns]
    return written != [str(name) for name in fitted] and [
        re.sub(r"\s", "_", name) for name in written
    ] == [str(name) for name in fitted]


#: Reference rows go up in pieces of about this many bytes.
REFERENCE_PIECE_BYTES = 32 * 1024 * 1024

#: Called as ``progress(stage, done, total)`` while a large model uploads.
Progress = Callable[[str, int, int], None]

#: Feature types whose rows are a table: one value for each feature.
_TABLE_TYPES = {"numeric", "integer", "binary", "categorical", "float", "number", "int", "bool", "boolean"}


#: The API release that runs model graphs (structures of ``family: "graph"``) and stores whole-number tensors. It
#: also knows the tasks, the feature types and the agents that came with them.
GRAPH_API = "1.3.0"
#: What an API from before it says when it is sent one: it reads a graph as a network, and knows three number types.
_OLD_API_REFUSALS = ("needs a layers array", "Unsupported tensor dtype")
#: The tasks and feature types an API from before it knows.
_OLD_TASKS = ("binary_classification", "multiclass_classification", "regression")
_OLD_FEATURE_TYPES = {"numeric", "integer", "binary", "categorical", "image"}


def _what_older_refused(exc: APIStatusError) -> Optional[str]:
    """What of an upload ``exc`` refuses the way an API from before :data:`GRAPH_API` refuses it: ``"graph"`` (a
    structure it reads as a network, or a tensor type it does not store), ``"task"`` (its schema has three),
    ``"feature"`` (it knows five feature types) or ``"inference"`` (it has no agents). ``None`` for any other."""
    said = str(exc.message or "")
    param = str(exc.param or "")
    if exc.status_code == 400 and any(sign in said for sign in _OLD_API_REFUSALS):
        return "graph"
    if exc.status_code == 422 and exc.code == "validation_error" and param in ("task", "inference.type"):
        return "task" if param == "task" else "inference"
    if exc.status_code == 400 and exc.code == "invalid_feature_schema" and "Unsupported feature type" in said:
        return "feature"
    return None


def _version(text: str) -> Tuple[int, ...]:
    """``"1.2.0"`` as ``(1, 2, 0)``; what is not a number counts as nothing."""
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = re.match(r"\d+", piece)
        parts.append(int(digits.group(0)) if digits else 0)
    return tuple(parts)


def _is_graph(spec: Mapping[str, Any]) -> bool:
    return str(spec.get("family") or "") == "graph"


def _reference_kind(spec: Mapping[str, Any], features: Sequence[Mapping[str, Any]]) -> str:
    """What reference rows for this model are: a ``table``, one ``image`` each, or ``other`` (text, a series, a tensor)."""
    if image_shape_of(spec) is not None:
        return "image"
    kinds = {str(feature.get("type") or "numeric").lower() for feature in features}
    return "table" if kinds <= _TABLE_TYPES else "other"


class Models(Resource):
    def create(
        self,
        *,
        name: str,
        task: Optional[str] = None,
        features: Optional[Sequence[Mapping[str, Any]]] = None,
        slug: Optional[str] = None,
        version: str = "1",
        description: Optional[str] = None,
        framework: str = "custom",
        outputs: Optional[Mapping[str, Any]] = None,
        inference: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        structure: Optional[Mapping[str, Any]] = None,
        project_id: Optional[str] = None,
        performance_metric: Optional[str] = None,
        **extra: Any,
    ):
        """Register a model. ``performance_metric`` (``"precision"`` or ``"recall"``, classifiers only) is what its
        performance checks watch instead of AUC or accuracy; see :meth:`set_performance_metric`."""
        body = {
            "name": name,
            "task": task,
            "features": [dict(f) for f in features] if features is not None else None,
            "slug": slug,
            "version": version,
            "description": description,
            "framework": framework,
            "outputs": dict(outputs) if outputs else None,
            "inference": dict(inference) if inference else None,
            "tags": list(tags) if tags is not None else None,
            "metadata": dict(metadata) if metadata else None,
            "structure": dict(structure) if structure else None,
            "project_id": project_id,
            "performance_metric": performance_metric,
        }
        body.update(extra)
        return self._post("/v1/models", compact(body))

    def retrieve(self, model: str) -> APIObject:
        return self._get(f"/v1/models/{model}")

    def list(
        self,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        task: Optional[str] = None,
        project_id: Optional[str] = None,
        unassigned: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/models",
            {
                "q": q,
                "status": status,
                "task": task,
                "project_id": project_id,
                "unassigned": unassigned,
                "limit": limit,
                "offset": offset,
            },
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        """Yield every model, following pagination automatically."""
        params.pop("offset", None)
        return self._iter("/v1/models", params)

    def update(self, model: str, **fields: Any):
        return self._patch(f"/v1/models/{model}", compact(fields))

    def delete(self, model: str):
        return self._delete(f"/v1/models/{model}")

    def assign_project(self, model: str, project_id: Optional[str]):
        """Move a model into a project. Pass ``None`` to leave it ungrouped."""
        return self._patch(f"/v1/models/{model}", {"project_id": project_id})

    def versions(self, model: str) -> Page:
        return self._page(f"/v1/models/{model}/versions")

    def create_version(
        self,
        model: str,
        *,
        version: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        slug: Optional[str] = None,
    ):
        return self._post(
            f"/v1/models/{model}/versions",
            compact({"version": version, "name": name, "description": description, "slug": slug}),
        )

    def delete_version(self, model: str, version: str):
        return self._delete(f"/v1/models/{model}/versions/{version}")

    def predict(self, model: str, *, input: Mapping[str, Any], log: bool = True, metadata: Optional[Mapping[str, Any]] = None):
        row = self._check_gaps(model, dict(input))
        return self._post(
            f"/v1/models/{model}/predict",
            compact({"input": row, "log": log, "metadata": dict(metadata) if metadata else None}),
        )

    def stats(self, model: str, *, days: int = 30):
        return self._get(f"/v1/models/{model}/stats", {"days": days})

    def performance(self, model: str, *, window: str = "week", days: int = 90):
        """Quality against the outcomes attached to logged predictions: overall, and per ``day`` / ``week`` / ``month``.

        ``overall`` holds the metrics (accuracy, precision, recall, F1, AUC, Brier, calibration and the confusion
        matrix for a binary model; macro F1 and per-class numbers for multiclass; MAE, RMSE, bias and R² for a
        regression); ``series`` the headline numbers per window; ``coverage`` the share of predictions with an outcome.
        """
        return self._get(f"/v1/models/{model}/performance", {"window": window, "days": days})

    def pr_curve(self, model: str, *, days: int = 90):
        """Precision against recall at every threshold, over the last ``days`` of outcomes; ``None`` without one.

        For a two-class model whose logged predictions carry a score. ``points`` (lowest threshold first) give the
        counts ``tp``, ``fp``, ``fn``, ``tn`` and the ``precision``, ``recall`` and ``f1`` when a row counts as
        positive at a score of at least ``threshold``; ``threshold`` is the model's own, ``prevalence`` the share of
        positives. ``pandas.DataFrame(curve.points)`` makes a table; :func:`kaimei.threshold_for` picks a threshold.
        The average precision is in ``performance(...).overall.average_precision``.
        """
        report = self.performance(model, days=days)
        return (report.get("overall") or {}).get("pr_curve")

    def set_performance_metric(self, model: str, metric: Optional[str]):
        """Have a classifier's performance checks watch ``"precision"`` or ``"recall"`` (their macro average over
        more than two classes) instead of AUC or accuracy; ``None`` goes back to the default.

        A degradation check and its alerts then compare that number with the outcomes before, as they did AUC.
        An API from before precision and recall ignores the field, so this raises :class:`KaimeiError` there
        rather than leave the checks watching what they did.
        """
        answer = self._patch(f"/v1/models/{model}", {"performance_metric": metric})
        if "performance_metric" not in answer:
            raise KaimeiError(
                "This Kaimei API does not know performance_metric, so nothing changed: its performance checks still "
                "watch AUC or accuracy. Upgrade the API to choose precision or recall."
            )
        return answer

    def monitoring(
        self,
        model: str,
        *,
        window: str = "day",
        days: int = 30,
        feature: Optional[str] = None,
        trailing: int = 500,
        points: int = 1500,
    ):
        """What was logged for a model over the last ``days``, and what a drift check would have found along it.

        ``series`` has one entry per ``hour`` / ``day`` / ``week``: the predictions logged in it, their mean output,
        and each input's PSI for the ``trailing`` rows that were the latest when it ended, with the same thresholds
        as the drift analysis (``status`` is ``stable``, ``moderate`` or ``significant``). ``levels`` is the PSI that
        many rows drawn from the reference data itself can give each input: at or under its own level, an input's
        drift cannot be told from sampling. ``top_feature`` is the input the window's reading comes from, ``chance``
        its level, and ``beyond_chance`` whether any input drifted. A window of few rows scores high by sampling
        alone, so read ``status`` together with it.
        ``features`` ranks the inputs by their highest PSI. ``tape`` draws one input in full (``feature``, or the most
        drifted): up to ``points`` logged predictions as ``[id, time, input value, output]``, the reference data's
        histogram and each window's. ``insights`` are the stored drift and performance insights in the range.

        Nothing is run and nothing is stored. Drift needs reference data and inputs that are single values (a table);
        without them ``drift_unavailable`` says why and the traffic is still returned.
        """
        return self._get(
            f"/v1/models/{model}/monitoring",
            compact({"window": window, "days": days, "feature": feature, "trailing": trailing, "points": points}),
        )

    def population(self, model: str, *, days: int = 30, limit: int = 2000):
        """A model's logged predictions one by one, and the ways they group.

        ``rows`` holds the newest ``limit`` predictions of the last ``days`` as columns (``id``, ``time``,
        ``value``, ``predicted``, ``actual``), and ``fields`` what they can be grouped by: every single-valued
        input, metadata keys logged beside the predictions that have few values or are numbers, what was
        predicted, what happened, and whether the two agree. Each field has its ``groups`` (label, rows, mean
        output, predicted and recorded classes) and ``separation``, the share of the variation in the model's
        answers that lies between its groups; the fields come most separating first.
        ``kaimei.display.rows(population)`` and ``population.to_frame()`` give one row per prediction with a
        column per field, ready for ``groupby``.

        Nothing is run and nothing is stored, so this works for any model, one Kaimei cannot run included; a
        model whose inputs are pictures, text or series is grouped by its metadata and its answers.
        """
        return self._get(f"/v1/models/{model}/population", {"days": days, "limit": limit})

    def similar(
        self,
        model: str,
        input: Optional[Mapping[str, Any]] = None,
        prediction_id: Optional[str] = None,
        *,
        k: int = 10,
        source: str = "reference",
    ):
        """The rows Kaimei already holds that are most like one case, and what was true of them.

        The case is ``input`` (its inputs) or ``prediction_id`` (a logged prediction); one of the two. It is
        compared with the model's reference rows (``source="reference"``) or its newest 2,000 logged
        predictions (``source="predictions"``), and the ``k`` nearest (50 at most) come back as ``cases``,
        nearest first. Each has ``id`` (a reference row's position, or a prediction's id), ``distance``,
        ``similarity`` (one less the distance), its ``inputs``, each input's own ``gaps`` and the inputs it
        ``differs`` on most, with both values. A reference row carries its ``label``; a logged prediction its
        ``output``, the class ``predicted``, its ``value`` and its recorded ``outcome``. ``summary`` counts
        those per class (or gives the mean and range of numbers) and says it in a ``sentence``: "Of the 10
        reference rows most like this case, 7 were labelled decline and 3 approve."
        ``kaimei.display.rows(similar)`` and ``similar.to_frame()`` give one row per case, with an
        ``input:<name>`` column per input.

        Two rows are compared input by input: a number by its gap in units of the width of the middle nine
        tenths of the reference rows' values (at most 1), a category or a yes/no by whether they match, and
        a value missing on one side only as far as can be. The distance is the mean over the inputs, each
        weighing the same. A value that is ``None`` or NaN is missing.

        Nothing is run and nothing is stored, so a key that only reads can call it. A model whose inputs are
        not all single values (a picture, a text, a series) answers ``422``.
        """
        if (input is None) == (prediction_id is None):
            raise ValueError("Pass either input (the case's inputs) or prediction_id (a logged prediction).")
        row = None
        if input is not None:
            # A gap in a table is NaN there and null on the wire: the value is missing.
            row = {str(key): (None if isinstance(value, float) and value != value else value) for key, value in dict(input).items()}
        return self._post(
            f"/v1/models/{model}/similar",
            compact({"input": row, "prediction_id": prediction_id, "k": k, "source": source}),
        )

    def holdings(self, model: str):
        """What Kaimei holds for a model and what stays with you, read from what is stored.

        ``runs`` says who runs the model (``by``: ``kaimei``, ``your_endpoint``, ``your_agent`` or ``nobody``)
        in a ``sentence``. ``items`` covers the weights and structure, the input and output schema, reference
        rows, logged predictions, recorded outcomes, explanations, insights and whether a credential for your
        endpoint is held (never its value); each has ``held`` (``yes``, ``partly`` or ``no``), a ``count`` in
        its ``unit``, the ``bytes`` stored where that is measured, and a ``detail``. ``crosses`` says what
        travels between your network and Kaimei ``when`` a prediction, an explanation or an analysis is asked
        for and when a prediction is logged. ``kaimei.display.rows(holdings)`` and ``holdings.to_frame()``
        give one row per item.

        Nothing is run and nothing is stored: counts and sizes are asked of the database.
        """
        return self._get(f"/v1/models/{model}/holdings")

    def case_study(self, model: str):
        """A model written up as a case study, from what Kaimei already holds about it.

        ``headline`` says what the model is, ``facts`` are the key numbers, and ``sections`` are always the same
        ten in order: ``overview``, ``drivers``, ``behaviour``, ``interactions``, ``example`` (one stored
        explanation, the one whose answer is nearest the median logged answer), ``outputs``, ``robustness``,
        ``population``, ``drift`` and ``performance``. Each has a ``summary`` of one to three sentences built
        from the numbers, a ``status`` (``ready``, ``stale`` when its insight is older than 30 days, or
        ``missing``), ``applies`` (false when this kind of model cannot have the section), a ``table`` of its key
        numbers and the ``data`` a chart needs. ``gaps`` lists what is not on file and what would add it.
        ``kaimei.display.rows(study)`` and ``study.to_frame()`` give one row per section.

        Nothing is run and nothing is stored: the model is not called and no analysis is started, so this
        works for any model and counts against no limit. ``case_study_markdown`` gives the same as text.
        """
        return self._get(f"/v1/models/{model}/case-study")

    def case_study_markdown(self, model: str) -> str:
        """The model's case study (see ``case_study``) as a Markdown document, ready for a wiki or a model card.

        A title, the facts as a table, a heading per section with its sentences and a table of its key numbers,
        and the list of what is not on file. It holds no HTML.
        """
        raw = self._client._request("GET", f"/v1/models/{model}/case-study", params={"format": "markdown"}, raw=True)
        return bytes(raw).decode("utf-8")

    def compare(self, model: str, other: str, *, source: str = "reference", limit: int = 500):
        """Run two models on the same rows: how far they agree, how each scores, and where they part.

        The rows are ``model``'s: its reference rows (``source="reference"``) or the inputs of its newest logged
        predictions (``source="predictions"``), ``limit`` of them at most. Everything given per model is a pair,
        ``model`` first: ``models``, ``scores`` (accuracy and AUC, or MAE and R², on the rows whose truth is
        known), and in ``rows`` the ``values`` and ``decisions``. ``agreement`` has ``rate`` (the share of rows
        decided the same way; classifiers only), ``correlation``, ``mean_gap`` and ``max_gap``; ``differ_most``
        lists the rows furthest apart; ``importance`` pairs the two models' global importance when both have it
        stored. ``kaimei.display.rows(comparison)`` and ``comparison.to_frame()`` give one row per row compared,
        with each model's value and decision, the ``gap`` and an ``input:<name>`` column per input.

        The two must take the same inputs and both predict a number or one of the same classes; otherwise the
        API answers ``422`` and says what differs. Both models are run and nothing is stored, so a key
        that only reads can call it (with ``source="predictions"`` it must be able to read predictions). It has
        hourly and daily allowances of its own and is refused once the month's analyses are used up.
        To see why the two answer a row as they do, explain that row with each (``client.explanations.create``).
        """
        return self._get(f"/v1/models/{model}/comparison", {"with": other, "source": source, "limit": limit})
    def evidence(self, model: str, *, fresh_days: Optional[int] = None) -> APIObject:
        """What Kaimei holds for a model against the requirements its guides cover, requirement by requirement.

        ``frameworks`` are the rules (the EU AI Act, adverse action reasons, model risk management, the GDPR's
        automated decisions); each requirement states what it asks, the ``evidence`` held toward it with its
        ``status`` (``held``, ``stale``, ``missing`` or ``not_applicable``), and what stays with you. ``todo``
        lists what would add the evidence that is missing. An analysis of live traffic older than ``fresh_days``
        (90 unless you say otherwise) is ``stale``.

        It is a check of what is stored. It does not say that a model, or its use, complies with anything.
        """
        return self._get(f"/v1/models/{model}/evidence", {"fresh_days": fresh_days})

    def report(self, model: str, *, fresh_days: Optional[int] = None) -> APIObject:
        """The model documentation report, as data: what the model is, what it reads and answers, the latest
        finding of each analysis in the analysis's own words, the data and records held, the reason table in
        force, its versions, and the evidence held by requirement. Nothing is computed for it."""
        return self._get(f"/v1/models/{model}/report", {"fresh_days": fresh_days})

    def report_pdf(self, model: str, *, paper: str = "letter", fresh_days: Optional[int] = None) -> bytes:
        """The model documentation report as a PDF, as bytes. ``paper`` is ``"letter"`` or ``"a4"``."""
        if paper not in ("letter", "a4"):
            raise ValueError("paper is 'letter' or 'a4'.")
        params = {"format": "pdf", "paper": paper, "fresh_days": fresh_days}
        return bytes(self._client._request("GET", f"/v1/models/{model}/report", params=params, raw=True))

    def save_report_pdf(
        self,
        model: str,
        path: Union[str, Path, None] = None,
        *,
        paper: str = "letter",
        fresh_days: Optional[int] = None,
    ) -> Path:
        """Save the model documentation report as a PDF and return where it was written.

        ``path`` is a file, or a folder to put ``model-documentation-<model>.pdf`` in. Left out, the file goes in
        the current folder under that name.
        """
        name = "model-documentation-" + (re.sub(r"[^A-Za-z0-9._-]+", "-", str(model)).strip("-.") or "model") + ".pdf"
        target = Path(path) if path is not None else Path(name)
        if target.is_dir():
            target = target / name
        target.write_bytes(self.report_pdf(model, paper=paper, fresh_days=fresh_days))
        return target

    def upload_reference_data(
        self,
        model: str,
        *,
        rows: Sequence[Mapping[str, Any]],
        labels: Optional[List[Any]] = None,
        mode: str = "replace",
    ):
        return self._put(
            f"/v1/models/{model}/reference-data",
            compact({"rows": [dict(r) for r in rows], "labels": labels, "mode": mode}),
        )

    def reference_data(self, model: str, *, sample: int = 5):
        """A summary of a model's reference data: how many ``rows`` it has, whether they have labels, and
        the first ``sample`` of them (up to 100) under ``sample``.

        ``sample_labels`` holds the label stored with each of those rows, in the same order, and ``None`` for a
        row stored without one, so ``zip(summary.sample, summary.sample_labels)`` pairs each row with what
        happened. An API before 1.5.0 sends no ``sample_labels``.
        """
        return self._get(f"/v1/models/{model}/reference-data", {"sample": sample})

    def delete_reference_data(self, model: str):
        return self._delete(f"/v1/models/{model}/reference-data")

    def upload(
        self,
        estimator: Any,
        *,
        name: str,
        feature_names: Optional[Sequence[str]] = None,
        features: Optional[Sequence[Mapping[str, Any]]] = None,
        task: Optional[str] = None,
        framework: Optional[str] = None,
        outputs: Optional[Mapping[str, Any]] = None,
        inference: Optional[Mapping[str, Any]] = None,
        slug: Optional[str] = None,
        version: str = "1",
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        project_id: Optional[str] = None,
        X: Any = None,
        y: Any = None,
        weights: str = "full",
        externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
        progress: Optional[Progress] = None,
        input_shape: Optional[Sequence[int]] = None,
        preprocess: Optional[Mapping[str, Any]] = None,
        value_range: Optional[Sequence[float]] = None,
        on_unsupported: str = "draw",
        via: str = "auto",
        sample: Any = None,
        inputs: Optional[Mapping[str, Any]] = None,
        output: Any = None,
        mask_layout: Optional[str] = None,
        threshold: Optional[float] = None,
        tokenizer: Any = None,
        labels: Optional[Sequence[Any]] = None,
        max_length: Optional[int] = None,
        save_onnx: Any = None,
        sampler: Any = None,
    ):
        """Register a fitted estimator and upload its layers and learned weights.

        For supported frameworks, the ONNX conversion route can upload a recurrent network, a transformer,
        a model with several inputs, or a scikit-learn estimator as a graph. A segmentation with
        ambiguous axes accepts ``mask_layout="chw"``, ``"hwc"`` or ``"indices"``. ``threshold`` says where the one
        score for each pixel of a segmentation is cut, in the model's own numbers; it is found from the model
        when not given, and applies only to a segmentation whose mask is one score for each pixel (a
        classifier's own threshold is ``outputs={"threshold": ...}``). ``via`` picks the route (``"auto"``,
        ``"native"`` or ``"onnx"``); an ONNX model itself (a ``.onnx`` path, bytes or ``ModelProto``) can
        be given in place of the estimator. ``sample`` is an example input for the conversion (required
        for token ids and for several inputs), ``inputs`` says what an input is, ``output`` which output
        the prediction is read from, ``tokenizer`` makes a token input a text feature, and ``labels``
        names the classes. ``task`` may also be ``text_generation``, ``generation`` or ``embedding``.
        See :func:`kaimei.structure.extract_structure`.

        A model that reads tokens is uploaded for one number of them, because a graph's shapes are fixed:
        as many as ``sample`` holds. That is the most tokens the model is ever given on Kaimei. A longer
        text is cut to it, and a language model's prompt and what it writes share it, so a model uploaded
        with a four-token sample stops writing after four tokens. Give a sample as long as the texts you
        will send, or pass ``max_length`` and the sample is filled up to it with the padding token.

        ``sampler`` makes a diffusion model's denoiser generate pictures with ``generate``: its ``diffusers``
        scheduler, or a dict of Kaimei's sampler settings. A ``diffusers`` pipeline may be given as the estimator.

        ``save_onnx`` is a path to write the converted ONNX model to, when the model was read through one.
        Nothing of it is uploaded beyond the structure: it is the file the model's agent runs to say what
        every step computed (``kaimei.agent.serve(client, model, onnx=path)``).

        A graph needs Kaimei API 1.3.0. An API from before it refuses one, and nothing is made: with
        ``via="auto"`` the model is then read by Kaimei's own readers, as clients before 0.12.0 read it
        (refused, or drawn and not run, where 1.3.0 would run it), with a warning; otherwise (another ``via``,
        an ONNX model, or ``inputs``, ``output``, ``tokenizer``, ``mask_layout`` or ``threshold``, which only a
        graph is read with) the error says which API it needs.

        A classifier that predicts with a threshold of its own (scikit-learn's ``TunedThresholdClassifierCV``
        and ``FixedThresholdClassifier``) carries it in its structure (``outputs.threshold``, in the units
        Kaimei reports, and the class it is for), and the API registers the model with it. A threshold or a
        ``positive_label`` given in ``outputs`` is your own decision instead: it is sent and kept as it is, with
        a warning when Kaimei's class will then not be the class the model's own ``predict`` gives.

        Image models: ``input_shape`` is required for a convolutional ``torch.nn.Module``, in
        PyTorch order ``(channels, height, width)``; Keras models report their own. ``preprocess``
        bakes the input normalization into the graph (``{"mean": [...], "std": [...]}`` on
        ``[0, 1]`` pixels, like torchvision's ``Normalize``). ``value_range=[0, 255]`` declares
        the pixel units you will send and the model reads. ``X`` may then be a batch of images,
        ``(n, h, w, c)`` or ``(n, c, h, w)``.

        Call this from a notebook that already holds the trained object. Kaimei
        stores the extracted spec (not the original ``.h5`` / ``.pt`` file) so the
        dashboard can draw every layer and tensor. Tensors with more than
        ``externalize_above`` values upload as compressed binary blobs instead of JSON,
        so models with millions of parameters work; ``weights="summary"`` sends shapes
        and statistics only and keeps the weights in your environment (a model of any size
        can be registered that way); ``weights="pictures"`` is a summary that also sends a
        small picture of each large tensor, which gives a tensor of 64 by 64 values or fewer away.

        Executable families (linear, MLP, trees and tree ensembles of any size, models
        made of those, dense/conv nets with every weight uploaded) can be queried for
        explanations and analyses.
        Summaries and approximate nets need ``inference={"type": "remote", "url": "..."}``
        or pre-computed attributions. ``progress(stage, done, total)`` is called while
        tensors upload.

        Layers and operations Kaimei cannot replay (attention, matmul, reshapes) are drawn
        from their shapes and attributes, and the model is then drawn rather than run;
        ``on_unsupported="error"`` refuses such models instead.
        """
        # A list from here on, and in what is asked again: names given as a generator can be read once.
        labels = label_names(labels)
        asked = {key: value for key, value in locals().items() if key != "self"}  # to ask again, by another route
        _check_onnx_path(save_onnx)  # before anything is made: a file that cannot be written is the caller's to fix
        if feature_names is None and features is None and _plainly_named(X, estimator):
            # Fitted on an array and given a table: the table is where the names are. A table of another width
            # is not the one it was fitted on, and is said to be so as X, not as names the caller never gave.
            # Only of a model fitted on a table: a network's own attribute of that name (``num_features``, the
            # width of a hidden layer, say) says nothing about the columns of X.
            a_network = (hasattr(estimator, "named_parameters") and hasattr(estimator, "forward")) or hasattr(
                estimator, "layers"
            )
            fitted_on = None if a_network else _input_width(estimator)
            if fitted_on is not None and fitted_on != len(X.columns):
                raise StructureError(
                    f"X has {len(X.columns)} columns, and this model was fitted on a table of {fitted_on}. X has to "
                    "be the table it was fitted on, with its columns in that order (or pass feature_names, a name "
                    "for each column it was fitted on).",
                    code="invalid_reference_rows",
                    param="X",
                )
            feature_names = [str(column) for column in X.columns]
        extraction = extract_structure(
            estimator,
            feature_names=feature_names,
            task=task,
            framework=framework,
            weights=weights,
            externalize_above=externalize_above,
            input_shape=input_shape,
            preprocess=preprocess,
            check_rows=X,
            on_unsupported=on_unsupported,
            via=via,
            sample=sample,
            inputs=inputs,
            output=output,
            mask_layout=mask_layout,
            threshold=threshold,
            tokenizer=tokenizer,
            labels=labels,
            max_length=max_length,
            sampler=sampler,
        )
        spec = extraction.spec
        warn_if_large_spec(spec)
        graph = str(spec.get("family") or "") == "graph"
        names = list(spec.get("feature_names") or []) if graph or feature_names is None else list(feature_names)
        feature_list = (
            [dict(item) for item in features]
            if features is not None
            else features_from_spec(spec, value_range=value_range)
        )
        # A converted graph's task was settled with the caller's as its starting point (an encoder-decoder said to be a
        # language model is one that writes from its input): the structure's is the model's.
        resolved_task = (spec.get("task") if graph else None) or task or spec.get("task")
        resolved_framework = framework or spec.get("framework") or detect_framework(estimator)
        # What the caller says of the model's outputs is sent as they say it. Left unsaid, the outputs are the
        # structure's, but for a decision of the model's own (its threshold, and the class that is for): that travels
        # in the structure, and the API registers the model with it by one rule, for this upload and for every
        # structure the model is given later. A decision the caller gives is the caller's own, and they are told
        # when it is not the model's.
        resolved_outputs = dict(outputs) if outputs else _undecided(spec.get("outputs"))
        if labels:
            resolved_outputs = {**dict(resolved_outputs or {}), "labels": [str(label) for label in labels]}
        own_decision = _says_a_decision(outputs)
        if own_decision:
            _say_when_the_decision_is_not_the_models(spec, resolved_outputs, whose="yours")
        resolved_inference = dict(inference) if inference else {"type": "none"}
        extra_meta: Dict[str, Any] = {"uploaded_from": "python-client"}
        if extraction.externalized:
            extra_meta["binary_tensors"] = len(extraction.tensors)
        if metadata:
            extra_meta.update(dict(metadata))
        # Binary tensors go up before the structure, so a spec with them holds its check until then.
        if resolved_inference.get("type", "none") == "none" and weights in ("summary", "pictures"):
            # Asked for: no weight leaves this environment, so nothing is there for Kaimei to run.
            warnings.warn(
                f"weights='{weights}' sent none of this model's weights, so Kaimei draws its architecture and does not "
                "run it. An agent can run the model where it is (kaimei.agent, with inference={'type': 'agent'}); "
                "a remote inference URL or pre-computed attributions also give explain/predict/analyse.",
                UserWarning,
                stacklevel=2,
            )
        elif resolved_inference.get("type", "none") == "none" and graph and spec.get("approximate"):
            warnings.warn(
                f"Kaimei draws this model and does not run it: {str(spec.get('summary') or '').strip()} "
                "A scoring endpoint or an agent (kaimei.agent) can run it.",
                UserWarning,
                stacklevel=2,
            )
        elif resolved_inference.get("type", "none") == "none" and not spec_is_executable(spec):
            warnings.warn(
                "This estimator was stored as a sketch or feature-importance summary. "
                "Kaimei can visualize it, but explain/predict/analyse need a remote "
                "inference URL or pre-computed attributions.",
                UserWarning,
                stacklevel=2,
            )
        # Rows that can't be read are refused here, before there's a model on the server to leave behind.
        reference = None
        if X is not None and graph and _reference_kind(spec, feature_list or []) == "other":
            # Reference rows are a table or pictures. Text, series and tensors have no rows to store.
            warnings.warn(
                "X was not uploaded as reference data: this model reads text, a series or a tensor, and reference "
                "rows are stored for tables and images.",
                UserWarning,
                stacklevel=2,
            )
        elif X is not None:
            image = image_shape_of(spec)
            table = X if image is not None else _columns_as_read(X, names, spec)
            rows, _names = rows_from_inputs(table, names or None, image_shape=image)
            targets = labels_from_target(y) if y is not None else None
            per_row = int(image[0] * image[1] * image[2]) if image else 1
            reference = cap_reference_rows(rows, targets, values_per_row=per_row)

        # A structure that references binary tensors can only be validated once they exist, so a
        # large model is registered from its shapes first and receives the real spec afterwards.
        initial = extraction.summary_spec() if extraction.externalized else spec
        created: Any = None
        try:
            created = self.create(
                name=name,
                task=resolved_task,
                features=feature_list or None,
                slug=slug,
                version=version,
                description=description,
                framework=resolved_framework,
                outputs=resolved_outputs,
                inference=resolved_inference,
                tags=tags,
                metadata=extra_meta,
                structure=initial,
                project_id=project_id,
            )
            model_ref = str(created.get("id") or created.get("slug") or name)
            if extraction.externalized:
                self._send_tensors(model_ref, extraction.tensors, progress=progress)
                if progress:
                    progress("structure", 0, 1)
                self._put(f"/v1/models/{model_ref}/structure", spec)
                if progress:
                    progress("structure", 1, 1)
        except APIStatusError as exc:
            # An API from before graphs refuses a graph, and the tasks, feature types and agents that came with them.
            what = _what_older_refused(exc)
            older = self._older_api(exc) if what is not None and (graph or what != "graph") else None
            if older is None or what is None:
                raise
            if created is not None:
                # A model that cannot hold its structure is not left behind half made.
                self._delete_quietly(str(created.get("id") or created.get("slug")))
            from ..graphs import is_onnx

            if what == "inference" or not graph:
                # Kaimei's own readers would be refused the same: what is new here is the caller's to change.
                raise self._needs_newer(exc, older, what, resolved_task, feature_list or []) from exc
            file = is_onnx(estimator)
            # What the caller said that only a conversion reads: Kaimei's own readers refuse a model described with it.
            only = [] if file else _only_converted(estimator, inputs, output, tokenizer, mask_layout, threshold, task)
            if via != "auto" or file or only:
                if what == "graph":
                    raise self._needs_graphs(exc, older, file=file, only=only) from exc
                raise self._needs_newer(exc, older, what, resolved_task, feature_list or []) from exc
            try:
                uploaded = self.upload(**{**asked, "via": "native", "save_onnx": None})
            except StructureError as native:
                raise StructureError(
                    f"Kaimei API {older} does not run model graphs, and this model can only be uploaded as one (API "
                    f"{GRAPH_API} runs it). Kaimei's own readers refuse it: {native.message}",
                    code=native.code or "unsupported_estimator",
                    param=native.param,
                ) from native
            warnings.warn(
                f"Kaimei API {older} does not run model graphs, and this model was read as one, which API {GRAPH_API} "
                "runs. It was uploaded as Kaimei's own readers read it instead, as clients before 0.12.0 did: see its "
                "structure for whether it runs or is drawn."
                + (
                    " save_onnx wrote no file: what was uploaded was not read from the ONNX conversion."
                    if save_onnx is not None
                    else ""
                ),
                UserWarning,
                stacklevel=2,
            )
            return uploaded
        if not own_decision and hasattr(created, "get"):
            # The API registered the model's own decision by its rule: said, should it have registered another.
            _say_when_the_decision_is_not_the_models(spec, created.get("outputs"), whose="registered", model=model_ref)
        checked = spec.get("check") is not None or (graph and spec_is_executable(spec))
        shown: Any = None
        if checked and resolved_inference.get("type", "none") == "none":
            shown = self.structure(model_ref, detail="summary")
            if shown.get("runnable") is False and on_unsupported == "error":
                # The API has no kernel for a step, or its own check did not reproduce the model: it only draws it.
                gone = self._delete_quietly(model_ref)
                raise StructureError(
                    f"Kaimei only draws this model and does not run it: {shown.get('summary') or ''} "
                    "on_unsupported='error' asked for a model it runs, so "
                    + (
                        f"the model was deleted again ({model_ref})."
                        if gone
                        else f"it is not wanted, and deleting it again failed: delete {model_ref} yourself."
                    )
                    + " Give it an inference endpoint or an agent (kaimei.agent) to run it, or upload it with "
                    "on_unsupported='draw' to keep the drawing.",
                    code="unsupported_layers",
                )
        # The structure read from the ONNX conversion is uploaded and kept: the file is the one its agent runs. A file
        # that cannot be written after all is warned of, with the model's name: the model is made, and what is left
        # of its upload (its reference rows) still goes up.
        _keep_onnx(extraction, save_onnx, model_ref)
        if reference is not None:
            rows, targets = reference
            refused = self._upload_reference_in_pieces(model_ref, rows, targets)
            if refused:
                warnings.warn(
                    f"{refused} of {len(rows)} reference rows don't fit the features' declared types, so they were left "
                    "out, and statistics describe the rest.",
                    stacklevel=2,
                )
        if shown is not None:
            if shown.get("runnable") is False:
                said = str(shown.get("summary") or "")
                marker = "Kaimei draws but does not run:"
                if marker in said:
                    # The graph holds operators the API has no way to run yet.
                    missing = said.split(marker, 1)[1].strip().rstrip(".")
                    warnings.warn(
                        f"Kaimei cannot run these operators of this model yet: {missing}. It is drawn, not run. "
                        "A scoring endpoint or an agent (kaimei.agent) can run it.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    from ..graphs import drawn_reason

                    reason = drawn_reason(spec)
                    warnings.warn(
                        f"Kaimei draws this model and does not run it. {reason}"
                        if reason
                        else "Run by Kaimei, this reading didn't give the model's own outputs, so it is drawn, not run. "
                        "A scoring endpoint or an agent (kaimei.agent) can run it.",
                        UserWarning,
                        stacklevel=2,
                    )
        return created

    def _delete_quietly(self, model: str) -> bool:
        """Delete a model an upload is not leaving behind. Whether it is gone; a failure is the caller's to say."""
        try:
            self.delete(model)
        except KaimeiError:
            return False
        return True

    def _older_api(self, exc: APIStatusError) -> Optional[str]:
        """The API's version, when ``exc`` is an API from before model graphs refusing what is new in them (see
        ``_what_older_refused``), as its ``/health`` says.

        ``None`` for any other refusal: one the API makes of a graph it does run is passed on as it is.
        """
        if _what_older_refused(exc) is None:
            return None
        try:
            version = str(self._client.health().get("version") or "")
        except KaimeiError:
            return None
        if not version or _version(version) >= _version(GRAPH_API):
            return None
        return version

    def _needs_graphs(
        self,
        exc: APIStatusError,
        older: str,
        what: str = "graph",
        *,
        file: bool = False,
        only: Sequence[str] = (),
        read_here: bool = True,
    ) -> APIStatusError:
        """``exc`` said plainly: what is new in this client, the API release that takes it, and what the caller can do.

        ``file`` says the model is an ONNX model, which is a graph whatever ``via`` says. ``only`` names the caller's
        arguments that only a conversion reads, which ``via='native'`` refuses. ``read_here`` is false for a
        structure the caller handed over already read, which no route reads again.
        """
        if what == "graph":
            needs = (
                f"Kaimei API {older} cannot take this model: it was read as a graph of ONNX operators "
                f"(family 'graph'), which needs API {GRAPH_API} or later. Upgrade the API"
            )
            if file:
                told = f"{needs}: an ONNX model is always read as a graph, whatever via says."
            elif not read_here:
                told = f"{needs}: this structure is a graph already."
            elif only:
                listed = " and ".join(only)
                told = (
                    f"{needs}. {listed} {'is' if len(only) == 1 else 'are'} read only from a model converted to a "
                    f"graph, so Kaimei's own readers were not asked: leave {listed} out and upload with via='native' "
                    "to read the model as clients before 0.12.0 did (those readers may refuse it or only draw it)."
                )
            else:
                told = (
                    f"{needs}, or upload with via='native' to read it as clients before 0.12.0 did (Kaimei's own "
                    "readers, which may refuse it or only draw it)."
                )
        else:
            told = (
                f"Kaimei API {older} stores tensors as f16, f32 or f64; whole numbers and flags ({what}) need API "
                f"{GRAPH_API} or later. Pass dtype='f32' to send them as numbers, as clients before 0.12.0 did."
            )
        return APIStatusError(
            f"{told} The API said: {exc.message}",
            type=exc.type,
            code=exc.code,
            param=exc.param,
            status_code=exc.status_code,
            details=exc.details,
        )

    def _needs_newer(
        self,
        exc: APIStatusError,
        older: str,
        refused: str,
        task: Optional[str],
        features: Sequence[Mapping[str, Any]],
    ) -> APIStatusError:
        """``exc`` said plainly, for a task, a feature type or an agent an API from before graphs does not know."""
        if refused == "task":
            told = (
                f"Kaimei API {older} does not know the task '{task}': it knows {', '.join(_OLD_TASKS)}. The tasks "
                f"added since need API {GRAPH_API} or later."
            )
        elif refused == "feature":
            kinds = sorted({str(item.get("type") or "numeric").lower() for item in features} - _OLD_FEATURE_TYPES)
            named = ", ".join(f"'{kind}'" for kind in kinds) or "of this model's features"
            told = (
                f"Kaimei API {older} does not know the feature type {named}: it knows "
                f"{', '.join(sorted(_OLD_FEATURE_TYPES))}. Text, tokens, series and tensors need API {GRAPH_API} or later."
            )
        else:
            told = f"Kaimei API {older} has no agents: a model an agent runs (inference type 'agent') needs API {GRAPH_API} or later."
        return APIStatusError(
            f"{told} Upgrade the API. The API said: {exc.message}",
            type=exc.type,
            code=exc.code,
            param=exc.param,
            status_code=exc.status_code,
            details=exc.details,
        )

    def _upload_reference_in_pieces(self, model: str, rows: List[Any], labels: Optional[List[Any]]) -> int:
        """Upload rows in pieces well under what the API and the proxies in front of it accept. Returns how many it refused."""
        if not rows:
            return 0
        step = max(1, REFERENCE_PIECE_BYTES // max(len(encode_json(rows[0])), 1))
        refused = 0
        for start in range(0, len(rows), step):
            stored = self.upload_reference_data(
                model,
                rows=rows[start : start + step],
                labels=labels[start : start + step] if labels is not None else None,
                mode="replace" if start == 0 else "append",
            )
            refused += int(stored.get("rejected") or 0) if hasattr(stored, "get") else 0
        return refused

    def structure(
        self,
        model: str,
        *,
        root: Optional[str] = None,
        depth: Optional[int] = None,
        detail: Optional[str] = None,
        include_values: Optional[bool] = None,
        max_inline: Optional[int] = None,
        tree: Optional[int] = None,
    ):
        """The architecture document. ``root`` / ``depth`` cut a huge model to one block."""
        params = {
            "root": root,
            "depth": depth,
            "detail": detail,
            "include_values": include_values,
            "max_inline": max_inline,
            "tree": tree,
        }
        return self._get(f"/v1/models/{model}/structure", compact(params))

    def upload_structure(
        self,
        model: str,
        estimator_or_spec: Any,
        *,
        feature_names: Optional[Sequence[str]] = None,
        task: Optional[str] = None,
        framework: Optional[str] = None,
        weights: str = "full",
        externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
        progress: Optional[Progress] = None,
        input_shape: Optional[Sequence[int]] = None,
        preprocess: Optional[Mapping[str, Any]] = None,
        on_unsupported: str = "draw",
        metadata: Optional[Mapping[str, Any]] = None,
        via: str = "auto",
        sample: Any = None,
        inputs: Optional[Mapping[str, Any]] = None,
        output: Any = None,
        mask_layout: Optional[str] = None,
        threshold: Optional[float] = None,
        tokenizer: Any = None,
        labels: Optional[Sequence[Any]] = None,
        max_length: Optional[int] = None,
        save_onnx: Any = None,
        sampler: Any = None,
    ):
        """Attach layers and weights to an existing model.

        Pass a fitted estimator, an ONNX model (a ``.onnx`` path, bytes or ``ModelProto``), an
        :class:`~kaimei.structure.Extraction`, or an already-extracted structure dict (whose ``$tensor``
        references must already be uploaded; a graph spec that still holds numpy arrays is finalized
        here). ``input_shape``, ``preprocess``, ``on_unsupported``, ``via``, ``sample``, ``inputs``,
        ``output``, ``tokenizer``, ``labels``, ``max_length``, ``save_onnx`` and ``threshold`` (where the one
        score for each pixel of a segmentation is cut: found from the model when not given, and only for a
        segmentation whose mask is one score for each pixel) are as for :meth:`upload`;
        ``metadata`` adds your own entries to what the spec records about where it came from.

        The model's features stay as they are: a structure that reads other features than the model has
        (text where it had a table) needs a model created for it, which :meth:`upload` does. The decision it is
        registered with (its threshold, and the class that is for) follows a structure that carries one, a
        classifier that predicts with a threshold of its own, by the API's rule: the registered decision follows
        when it is the one every model starts with or the one the structure being replaced carried, and a
        decision of your own is kept. The model is read back, and you are warned when its registered decision is
        not the structure's; nothing but the structure is written.

        A graph (``family: "graph"``) needs Kaimei API 1.3.0. An API from before it refuses one, the model keeps
        the structure it had, and the error says which API is needed.
        """
        labels = label_names(labels)  # a list, before anything reads names given as a generator
        _check_onnx_path(save_onnx)  # before anything is sent: a file that cannot be written is the caller's to fix
        read_here = not isinstance(estimator_or_spec, (Extraction, Mapping))
        if isinstance(estimator_or_spec, Extraction):
            extraction = estimator_or_spec
        elif isinstance(estimator_or_spec, Mapping):
            from ..graphs import holds_arrays

            extraction = Extraction(spec=dict(estimator_or_spec))
            if str(extraction.spec.get("family") or "") == "graph" and holds_arrays(extraction.spec):
                extraction = finalize_spec(extraction.spec, weights=weights, externalize_above=externalize_above)
        else:
            extraction = None  # type: ignore[assignment]
        if extraction is not None:
            if metadata:
                merged = {**dict(extraction.spec.get("metadata") or {}), **{str(key): value for key, value in metadata.items()}}
                # Everything else it holds stays: its tensors, and the ONNX model it was read from (``save_onnx``).
                extraction = dataclasses.replace(extraction, spec={**extraction.spec, "metadata": merged})
        else:
            extraction = extract_structure(
                estimator_or_spec,
                feature_names=feature_names,
                task=task,
                framework=framework,
                weights=weights,
                externalize_above=externalize_above,
                input_shape=input_shape,
                preprocess=preprocess,
                on_unsupported=on_unsupported,
                metadata=metadata,
                via=via,
                sample=sample,
                inputs=inputs,
                output=output,
                mask_layout=mask_layout,
                threshold=threshold,
                tokenizer=tokenizer,
                labels=labels,
                max_length=max_length,
                sampler=sampler,
            )
        try:
            if extraction.externalized:
                self._send_tensors(model, extraction.tensors, progress=progress)
            stored = self._put(f"/v1/models/{model}/structure", extraction.spec)
        except APIStatusError as exc:
            what = _what_older_refused(exc)
            older = self._older_api(exc) if what is not None and (_is_graph(extraction.spec) or what != "graph") else None
            if older is None or what is None:
                raise
            # The model keeps the structure it had. Tensors sent before the refusal are stored with it, unused.
            if what != "graph":
                # A task, a feature type or an agent that API does not know: no other route would make it known.
                described = [dict(item) for item in extraction.spec.get("features") or [] if isinstance(item, Mapping)]
                raise self._needs_newer(exc, older, what, str(extraction.spec.get("task") or task or ""), described) from exc
            from ..graphs import is_onnx

            file = read_here and is_onnx(estimator_or_spec)
            only = (
                _only_converted(estimator_or_spec, inputs, output, tokenizer, mask_layout, threshold, task)
                if read_here and not file
                else []
            )
            raise self._needs_graphs(exc, older, file=file, only=only, read_here=read_here) from exc
        if _carried_decision(extraction.spec) is not None:
            # The API makes the model's registered decision follow this structure's by its own rule, and keeps one
            # that is the customer's own. Nothing is written here: the model is read back, and said when its
            # registered decision is not the one this structure's check compared.
            try:
                registered = self.retrieve(model).get("outputs")
            except KaimeiError as exc:
                warnings.warn(
                    f"This structure's class for a row goes by the model's own decision, and what model {model} is "
                    f"registered with could not be read back ({exc}): see client.models.retrieve({model!r}).outputs.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                _say_when_the_decision_is_not_the_models(extraction.spec, registered, whose="registered", model=model)
        # Written once the structure read from it is the model's: the file is the one its agent runs.
        _keep_onnx(extraction, save_onnx, model)
        return stored

    def upload_tensors(
        self,
        model: str,
        tensors: Mapping[str, Any],
        *,
        dtype: Optional[str] = None,
        progress: Optional[Progress] = None,
    ):
        """Upload weight tensors as compressed binary; the spec references them as ``{"$tensor": key}``.

        Each tensor goes in its own number type unless ``dtype`` says: ``f64`` when it is double precision,
        ``f16`` when half and ``f32`` for any other floats; whole numbers and flags as ``i8``, ``i16``,
        ``i32``, ``i64``, ``u8`` or ``bool``, which a model graph keeps for its index tables and masks.
        Whole numbers and flags need Kaimei API 1.3.0; ``dtype="f32"`` sends them as clients before 0.12.0 did.
        """
        try:
            return self._send_tensors(model, tensors, dtype=dtype, progress=progress)
        except APIStatusError as exc:
            older = self._older_api(exc)
            if older is None:
                raise
            kinds = sorted({dtype or tensor_dtype(array) for array in tensors.values()} - {"f16", "f32", "f64"})
            raise self._needs_graphs(exc, older, ", ".join(kinds) or "these tensors") from exc

    def _send_tensors(
        self,
        model: str,
        tensors: Mapping[str, Any],
        *,
        dtype: Optional[str] = None,
        progress: Optional[Progress] = None,
    ) -> List[Any]:
        items = list(tensors.items())
        results = []
        for index, (key, array) in enumerate(items):
            if progress:
                progress("tensors", index, len(items))
            shape = ",".join(str(int(dim)) for dim in getattr(array, "shape", ()) or (len(array),))
            kind = dtype or tensor_dtype(array)
            results.append(
                self._client._request(
                    "PUT",
                    f"/v1/models/{model}/structure/tensors/{key}",
                    params={"shape": shape, "dtype": kind, "compression": "zlib"},
                    content=encode_tensor(array, kind),
                )
            )
        if progress:
            progress("tensors", len(items), len(items))
        return results

    def tensors(self, model: str):
        """Binary tensors uploaded for this model, with shapes, sizes and statistics."""
        return self._get(f"/v1/models/{model}/structure/tensors")

    def delete_tensor(self, model: str, key: str):
        return self._delete(f"/v1/models/{model}/structure/tensors/{key}")

    def delete_structure(self, model: str):
        return self._delete(f"/v1/models/{model}/structure")

    def trace_structure(
        self,
        model: str,
        *,
        input: Mapping[str, Any],
        detail: Optional[str] = None,
        nodes: Optional[Sequence[str]] = None,
        tree: Optional[int] = None,
    ):
        """Replay one prediction. ``detail="summary"`` returns per-node statistics for large models."""
        row = self._check_gaps(model, dict(input))
        body = {"input": row, "detail": detail, "nodes": list(nodes) if nodes is not None else None, "tree": tree}
        return self._post(f"/v1/models/{model}/structure/trace", compact(body))

    def activations(
        self,
        model: str,
        *,
        input: Mapping[str, Any],
        nodes: Optional[Sequence[str]] = None,
        channels: Optional[int] = None,
        max_side: Optional[int] = None,
        units: Optional[int] = None,
        cam: Optional[Sequence[str]] = None,
        target: Optional[Union[int, str]] = None,
        channel_start: Optional[int] = None,
    ):
        """What every layer computed for one input.

        Every stage reports its intensity, active share and strongest channels or units. The
        detailed stages (``nodes``; default: up to 24 feature maps plus the output) also carry
        the ``Σ|a|`` energy of each channel, the maps of the ``channels`` strongest ones (at
        most ``max_side`` cells on a side), a saliency map and, for vectors up to 4,096 units,
        every value. ``cam`` names layers for class attention (Grad-CAM) toward ``target``.
        Decode the byte-packed maps with :mod:`kaimei.activations`.

        ``channel_start`` sends the maps from that channel on, in their own order, where the strongest are
        sent without it: the way to read every head of an attention layer, ``channels`` at a time. It is
        sent only when given, and needs Kaimei API 1.3.0.
        """
        if channel_start is not None and (isinstance(channel_start, bool) or not 0 <= int(channel_start) <= 1_000_000):
            raise ValueError("channel_start is the first channel whose map is sent: a whole number from 0.")
        body = {
            "input": dict(input),
            "nodes": list(nodes) if nodes is not None else None,
            "channels": channels,
            "max_side": max_side,
            "units": units,
            "cam": list(cam) if cam is not None else None,
            "target": target,
            "channel_start": int(channel_start) if channel_start is not None else None,
        }
        return self._post(f"/v1/models/{model}/structure/activations", compact(body))

    def structure_parameter(
        self,
        model: str,
        parameter_id: str,
        *,
        offset: int = 0,
        limit: int = 4096,
        stats: bool = False,
        preview: int = 0,
    ):
        return self._get(
            f"/v1/models/{model}/structure/parameters/{parameter_id}",
            compact({"offset": offset, "limit": limit, "stats": stats or None, "preview": preview or None}),
        )

    def parameter_view(
        self,
        model: str,
        parameter: str,
        *,
        side: int = 64,
        rows: Optional[Tuple[int, Optional[int]]] = None,
        cols: Optional[Tuple[int, Optional[int]]] = None,
        pool: str = "strongest",
    ):
        """A picture of one tensor as a matrix, at most ``side`` cells a side, whatever its size.

        The tensor's first axis runs down and the rest across. Each cell stands for a block of values: the
        strongest of them with its sign (``pool="strongest"``) or their mean (``"mean"``). ``rows`` and
        ``cols`` are ``(first, last)`` windows to zoom into (``last`` may be ``None`` for the end), down
        to one value a cell. ``view`` holds the cells as bytes with their range: decode it with
        :func:`kaimei.activations.decode`. The values themselves stay on the server.
        """
        if isinstance(side, bool) or not 1 <= int(side) <= 512:
            raise ValueError("side is how many cells the picture has on a side: 1 to 512.")
        return self._get(
            f"/v1/models/{model}/structure/parameters/{parameter}",
            compact({"side": int(side), "rows": _window(rows, "rows"), "cols": _window(cols, "cols"), "pool": pool}),
        )

    def generate(
        self,
        model: str,
        *,
        input: Optional[Mapping[str, Any]] = None,
        steps: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_tokens: Optional[Sequence[int]] = None,
        seed: int = 0,
        seeds: Optional[Sequence[int]] = None,
        draw: Optional[Sequence[str]] = None,
    ):
        """Run a generative model's loop: a language model writing token by token, a diffusion model denoising.

        ``input`` is the prompt for a language model (``{"prompt": "the film was"}``) and any conditioning
        for a model that draws; a tensor feature left out (a latent, the noise to start from) is given its
        baseline, or drawn from ``seed`` when it has none. ``steps`` is how many tokens to write or denoising
        steps to take. ``temperature`` 0 always takes the most likely token; above it tokens are drawn, among
        the ``top_k`` most likely and those that together hold ``top_p`` of the probability. ``stop_tokens``
        are token ids that end the text.

        ``seeds`` (1 to 16 of them) draws a model that answers in one pass (a generator, a decoder) once for
        each, returned as ``variations``: one prediction, however many seeds. ``draw`` names tensor or
        sequence features to draw from the seed although they have a baseline. Both are sent only when given.
        """
        if seeds is not None and (isinstance(seeds, (str, bytes)) or not 1 <= len(seeds) <= 16):
            raise ValueError("seeds is a list of 1 to 16 whole numbers, one draw for each.")
        if draw is not None and isinstance(draw, (str, bytes)):
            raise ValueError("draw is a list of feature names, not one name: draw=['latent'].")
        body = {
            "input": dict(input) if input else {},
            "steps": steps,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_tokens": [int(token) for token in stop_tokens] if stop_tokens is not None else None,
            "seed": seed,
            "seeds": [int(item) for item in seeds] if seeds is not None else None,
            "draw": [str(name) for name in draw] if draw is not None else None,
        }
        return self._post(f"/v1/models/{model}/generate", compact(body))

    def tokenize(
        self,
        model: str,
        *,
        text: Optional[str] = None,
        ids: Optional[Sequence[int]] = None,
        feature: Optional[str] = None,
    ):
        """Split ``text`` the way the model's tokenizer does, or name the tokens behind ``ids``.

        ``feature`` picks the text feature when the model has several. Text needs the model's tokenizer,
        which is uploaded with its structure (``tokenizer=`` on :meth:`upload`).
        """
        if (text is None) == (ids is None):
            raise ValueError("Give text to split, or ids to name: one of the two.")
        body = {"text": text, "ids": [int(index) for index in ids] if ids is not None else None, "feature": feature}
        return self._post(f"/v1/models/{model}/tokenize", compact(body))

    def agent_status(self, model: str):
        """Whether an agent is connected for this model right now, and what it answers (see :mod:`kaimei.agent`)."""
        return self._get(f"/v1/models/{model}/agent")

    def structure_parameter_values(self, model: str, parameter_id: str, *, offset: int = 0, limit: int = 1_000_000):
        """A window of one tensor as a float32 numpy array, fetched as binary."""
        import numpy

        data = self._client._request(
            "GET",
            f"/v1/models/{model}/structure/parameters/{parameter_id}",
            params={"format": "f32", "offset": offset, "limit": limit},
            raw=True,
        )
        return numpy.frombuffer(data, dtype="<f4").astype(numpy.float32)

    def transfer(
        self,
        model: str,
        *,
        organization: str,
        project: Optional[str] = None,
        include_versions: bool = True,
    ):
        """Move a model, with every version and all of its history, to another organization you belong to.

        ``project`` puts it in one of the destination's projects. Needs a session (``auth.login``) with an
        owner or admin role where the model is now. The result lists the models as they are in the destination.
        With Kaimei API 1.2.0 or newer, credentials saved for the model's own endpoint (its headers, and any in
        its URL) are removed, not moved (``credentials_removed`` lists the models that need them saved again), and
        the destination needs room on its plan for this month's predictions, explanations and analyses that come
        along.
        """
        return self._post(
            f"/v1/models/{model}/transfer",
            compact({"organization": organization, "project": project, "include_versions": include_versions}),
        )

    def fork(
        self,
        model: str,
        *,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        include_versions: bool = True,
    ):
        """Copy a model. Without ``organization`` the copy goes to your personal workspace, so name another
        organization when the client is working in the personal workspace itself.

        The copy holds the model's settings, versions, structure, weights and reference data. Predictions,
        explanations, analyses and insights stay with the original. Credentials for the model's own
        endpoint (its headers, and with Kaimei API 1.2.0 any in its URL) are not copied: ``credentials_removed``
        lists the copies that need them saved again.
        Needs a session (``auth.login``).
        """
        return self._post(
            f"/v1/models/{model}/fork",
            compact(
                {
                    "organization": organization,
                    "project": project,
                    "name": name,
                    "include_versions": include_versions,
                }
            ),
        )


def _check_onnx_path(path: Any) -> None:
    """Refuse a ``save_onnx`` that cannot be written, before anything is made on the server.

    The file is written once the structure read from it is the model's, which is after the model is made. A path
    that could never be written (a folder that is not there, a folder in place of a file, a file or folder that
    cannot be written to) is refused first, so that nothing is left half uploaded.
    """
    if path is None:
        return
    try:
        target = os.path.abspath(os.fspath(path))
    except TypeError:
        raise StructureError(
            "save_onnx is the path of the file to write the converted ONNX model to.",
            code="invalid_structure",
            param="save_onnx",
        ) from None
    folder = os.path.dirname(target)
    if os.path.isdir(target):
        why = "is a folder, not a file"
    elif not os.path.isdir(folder):
        why = f"is in a folder that does not exist ({folder})"
    elif os.path.exists(target):
        # The file is there already: whether it can be written is its own to say, whatever its folder allows.
        if os.access(target, os.W_OK):
            return
        why = "cannot be written to"
    elif not os.access(folder, os.W_OK):
        why = f"is in a folder that cannot be written to ({folder})"
    else:
        return
    raise StructureError(
        f"save_onnx is '{os.fspath(path)}', which {why}. Nothing was uploaded: give a path that can be written, or "
        "leave save_onnx out.",
        code="invalid_structure",
        param="save_onnx",
    )


def _keep_onnx(extraction: Extraction, path: Any, model: Optional[str] = None) -> None:
    """Write the ONNX model a structure was read from to ``path``, for the model's agent to run.

    By now the structure is the model's, so a file that cannot be written is warned of and not raised: ``model``
    names the model that was uploaded all the same.
    """
    if path is None:
        return
    data = getattr(extraction, "onnx", None)
    if data is None:
        warnings.warn(
            "save_onnx wrote no file: this structure was not read from an ONNX conversion made here (Kaimei's own "
            "reader read the model, or you gave an ONNX model, which is then the file to run).",
            UserWarning,
            stacklevel=3,
        )
        return
    try:
        with open(os.fspath(path), "wb") as handle:
            handle.write(data)
    except OSError as exc:
        named = f"model {model}" if model else "the model"
        warnings.warn(
            f"save_onnx could not write '{os.fspath(path)}' ({type(exc).__name__}: {exc}). The structure of {named} "
            "is uploaded all the same, and the ONNX model it was read from was not kept: upload the structure again "
            f"with client.models.upload_structure({(model or '<model>')!r}, <the model>, save_onnx=<a path that can be "
            "written>) to have its file.",
            UserWarning,
            stacklevel=3,
        )


#: Tasks whose structure can carry a decision of the model's own: where its two classes, or each of its labels, are cut.
_DECIDING_TASKS = ("binary_classification", "multilabel_classification")
#: What says a decision in a model's ``outputs``.
_DECISION_KEYS = ("threshold", "positive_label")


def _is_a_threshold(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _carried_decision(spec: Mapping[str, Any]) -> Optional[Tuple[float, int]]:
    """``(threshold, position of the positive class)`` a structure that runs carries for its model, or ``None``.

    A structure read from a classifier that predicts with a threshold of its own (scikit-learn's
    ``TunedThresholdClassifierCV`` and ``FixedThresholdClassifier``) carries it in ``outputs.threshold``, a number
    from 0 to 1, with ``outputs.positive_label`` when the class it is for is not the second: one of the structure's
    own class names. That is the decision its check compared with the model's own ``predict``, and the one the API
    registers the model with unless the customer gave another.

    Only a classifier's: where a segmentation's one score for each pixel is cut is its ``head.threshold``, which its
    structure keeps and no model is registered with.
    """
    outputs = spec.get("outputs")
    task = str(spec.get("task") or "")
    if task not in _DECIDING_TASKS or spec.get("approximate") or not isinstance(outputs, Mapping):
        return None
    if not _is_a_threshold(outputs.get("threshold")):
        return None
    positive = outputs.get("positive_label")
    if positive is None:
        return float(outputs["threshold"]), 1
    names = [str(label) for label in outputs.get("labels") or spec.get("classes") or spec.get("labels") or []]
    if task != "binary_classification" or len(names) != 2 or str(positive) not in names:
        return None  # no decision the API's rule follows
    return float(outputs["threshold"]), names.index(str(positive))


def _undecided(outputs: Any) -> Optional[Dict[str, Any]]:
    """A structure's ``outputs`` as a model's, without the decision they carry: the API registers that itself."""
    if not isinstance(outputs, Mapping):
        return None
    return {key: value for key, value in outputs.items() if key not in _DECISION_KEYS} or None


def _says_a_decision(outputs: Any) -> bool:
    """Whether a caller's ``outputs`` say where the model's classes are cut, or which class that is for."""
    return isinstance(outputs, Mapping) and any(outputs.get(key) is not None for key in _DECISION_KEYS)


def _registered_decision(outputs: Mapping[str, Any], names: Sequence[str]) -> Tuple[float, int]:
    """``(threshold, position of the positive class)`` a yes-or-no model with these ``outputs`` predicts by: one half
    and its second class, unless they say. ``names`` are its classes when ``outputs`` do not name them."""
    labels = [str(label) for label in outputs.get("labels") or names]
    threshold = float(outputs["threshold"]) if _is_a_threshold(outputs.get("threshold")) else 0.5
    positive = outputs.get("positive_label")
    position = labels.index(str(positive)) if positive is not None and str(positive) in labels[:2] else 1
    return threshold, position


def _decided_otherwise(spec: Mapping[str, Any], decision: Tuple[float, int]) -> Optional[Tuple[int, int]]:
    """``(on how many rows of the structure's check a model deciding by ``decision`` gives another class than the
    model's own, of how many)``, or ``None`` when the structure has no check. The model's own class on those rows is
    the one the structure's own decision gives: that is what its check compared."""
    from ..graphs import reported_class

    carried = _carried_decision(spec)
    check = spec.get("check")
    rows = check.get("outputs") if isinstance(check, Mapping) else None
    if carried is None or not isinstance(rows, (list, tuple)) or not rows:
        return None
    task = str(spec.get("task") or "")
    names = [str(label) for label in (spec.get("outputs") or {}).get("labels") or spec.get("classes") or spec.get("labels") or []]
    names = names if len(names) == 2 else ["0", "1"]

    def by(deciding: Tuple[float, int]) -> Dict[str, Any]:
        said: Dict[str, Any] = {"labels": names, "threshold": deciding[0]}
        if task == "binary_classification":
            said["positive_label"] = names[deciding[1]]
        return {"task": task, "outputs": said}

    def cls(found: Any) -> Any:
        return found.tolist() if hasattr(found, "tolist") else found

    try:
        differ = sum(1 for values in rows if cls(reported_class(by(carried), values)) != cls(reported_class(by(decision), values)))
    except (TypeError, ValueError):
        return None
    return int(differ), len(rows)


def _say_when_the_decision_is_not_the_models(
    spec: Mapping[str, Any], outputs: Any, *, whose: str, model: Optional[str] = None
) -> None:
    """Warn when the decision a model will predict by (``outputs``: the caller's own for an upload, or what the API
    has registered) is not the one its structure carries, which is the model's own.

    The structure's check compared Kaimei's class with the model's ``predict`` by the model's own decision. The API
    reports a class by the model's registered ``outputs``: a decision of the customer's own is theirs to choose and
    is kept, and Kaimei's class is then not always the class ``predict`` gives. Classes are told apart by their
    place, so the two may call them by different names.
    """
    carried = _carried_decision(spec)
    if carried is None or not isinstance(outputs, Mapping):
        return
    own = [str(label) for label in (spec.get("outputs") or {}).get("labels") or spec.get("classes") or spec.get("labels") or []]
    theirs = _registered_decision(outputs, own)
    if str(spec.get("task") or "") != "binary_classification":
        theirs = (theirs[0], carried[1])  # each label is cut on its own: there is no positive class to differ
    counted = _decided_otherwise(spec, theirs)
    # A threshold of exactly one half is carried as the largest number below it, so that an even split is the model's
    # positive class and not the first. The API takes the two for one, and they differ only on a row split evenly.
    close = abs(theirs[0] - carried[0]) <= 1e-12 and theirs[1] == carried[1]
    if close and (theirs[0] == carried[0] or counted is None or not counted[0]):
        return
    labels = [str(label) for label in outputs.get("labels") or own]

    def worded(decision: Tuple[float, int]) -> str:
        named = f"'{labels[decision[1]]}' " if len(labels) == 2 and str(spec.get("task")) == "binary_classification" else ""
        return f"{named}at {decision[0]!r} or more" if close else f"{named}at {decision[0]:g} or more"

    seen = f" (on {counted[0]:,} of the {counted[1]:,} rows it was checked on)" if counted is not None else ""
    wanted: Dict[str, Any] = {"threshold": carried[0]}
    if len(labels) == 2 and str(spec.get("task")) == "binary_classification":
        wanted["positive_label"] = labels[carried[1]]
    if whose == "yours":
        said = (
            f"outputs says this model predicts {worded(theirs)}, and the model itself predicts {worded(carried)} (in "
            f"the units Kaimei reports). Kaimei reports its class by yours, so it is not always the class predict() "
            f"gives{seen}. Leave threshold and positive_label out of outputs to register the model's own."
        )
    else:
        said = (
            f"Model {model} is registered to predict {worded(theirs)}, and the model this structure was read from "
            f"predicts {worded(carried)} (in the units Kaimei reports). A decision registered as the customer's own is "
            f"kept, and Kaimei reports its class by it, so it is not always the class predict() gives{seen}: "
            f"client.models.update({model!r}, outputs={wanted!r}) registers the model's own."
        )
    warnings.warn(said, UserWarning, stacklevel=3)


def _only_converted(
    estimator: Any, inputs: Any, output: Any, tokenizer: Any, mask_layout: Any, threshold: Any = None, task: Any = None
) -> List[str]:
    """The caller's arguments that only a model converted to a graph is read with, by name (see
    ``structure._onnx_only_arguments``): with any of them Kaimei's own readers refuse the model."""
    from ..structure import _onnx_only_arguments

    found = _onnx_only_arguments(estimator, inputs, output, tokenizer, task)
    return [*found, *(["mask_layout"] if mask_layout is not None else []), *(["threshold"] if threshold is not None else [])]


def _columns_as_read(X: Any, names: Sequence[str], spec: Mapping[str, Any]) -> Any:
    """``X`` as the table of the structure's features, when its columns are not that table already.

    A table pipeline lists its features in the order its parts read their columns, and leaves out the columns it
    drops, so the columns of the table it was fitted on do not line up with its features one for one. Its structure
    says where each feature is read from (``metadata.table_columns``: for each feature, in order, the position of
    its column in the fitted table). A table that names the features is read by those names, as any is. An array,
    or a table with other names, is read by those positions; one too narrow to hold them is refused.
    """
    positions = (spec.get("metadata") or {}).get("table_columns") if isinstance(spec.get("metadata"), Mapping) else None
    if not isinstance(positions, (list, tuple)) or not names or len(positions) != len(names):
        return X
    if not all(isinstance(position, numbers.Integral) and not isinstance(position, bool) for position in positions):
        return X
    positions = [int(position) for position in positions]
    if isinstance(X, list) and X and isinstance(X[0], Mapping):
        return X  # rows that name their values
    columns = getattr(X, "columns", None)
    if columns is not None and hasattr(X, "iloc"):
        if {str(name) for name in names} <= {str(column) for column in columns}:
            return X
        table = X.iloc[:, [position for position in positions if 0 <= position < len(columns)]].to_numpy(dtype=object)
        width = len(columns)
    else:
        import numpy

        table = numpy.asarray(X)
        if table.ndim != 2:
            return X  # not a table of rows: said where its rows are read
        width = int(table.shape[1])
        if all(0 <= position < width for position in positions):
            table = table[:, list(positions)]
    beyond = [(name, position) for name, position in zip(names, positions) if not 0 <= position < width]
    if beyond:
        name, position = beyond[0]
        raise StructureError(
            f"X has {width} column{'s' if width != 1 else ''}, and this model reads its feature '{name}' from column "
            f"{position} (counting from 0) of the table it was fitted on. X has to be that table, with its columns in "
            "that order.",
            code="invalid_reference_rows",
            param="X",
        )
    return table


def _window(span: Optional[Tuple[int, Optional[int]]], name: str) -> Optional[str]:
    """``(first, last)`` as the ``first:last`` the API reads, or ``None`` for everything."""
    if span is None:
        return None
    try:
        first, last = span
        first = int(first)
        last = None if last is None else int(last)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is a (first, last) pair of whole numbers; last may be None.") from exc
    if first < 0 or (last is not None and last <= first):
        raise ValueError(f"{name} is a (first, last) pair with last after first.")
    return f"{first}:" if last is None else f"{first}:{last}"
