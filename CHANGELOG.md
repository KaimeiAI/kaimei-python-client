# Changelog

## 0.3.0 — 2026-10-23

Models with millions of parameters and thousands of layers upload and render.

### Added

- Tensors with more than `externalize_above` values (default 65,536) leave the JSON spec and
  upload as zlib-compressed float32 blobs (`PUT /structure/tensors/{key}`), referenced from the
  spec as `{"$tensor": key}`. `client.models.upload()` registers the model from its shapes,
  streams the tensors, then sends the real spec; `progress=` reports each stage.
- `weights="summary"` uploads shapes and statistics only. The dashboard draws the architecture
  while the weights stay in your environment.
- `extract_structure()` returns an `Extraction` (spec plus tensors); `estimator_to_spec()` is
  unchanged and keeps everything inline.
- PyTorch and Keras layers carry their module `path`, so the API groups them into blocks and
  detects repeated blocks (`×N`).
- `client.models.tensors()`, `delete_tensor()`, `structure_parameter_values()` (binary window),
  `structure(root=, depth=, detail=)` and `trace_structure(detail="summary", nodes=)`.

## 0.2.2 — 2026-10-13

### Fixed

- mypy type-checks as Python 3.10 and resolves feature names to `list[str]` before
  reading a wrapped estimator.

## 0.2.1 — 2026-10-13

### Added

- `estimator_to_spec` unwraps wrappers around a fitted sklearn / Keras / Torch estimator
  (`model_`, `estimator_`, `best_estimator_`, `module`, and the same family) without
  special-casing vendor packages. Framework labels follow the core-library MRO.

## 0.2.0 — 2026-10-12

First release that can upload a fitted estimator from a notebook.

### Added

- `client.models.upload()` extracts sklearn / Keras / PyTorch structure and registers it in one call.
- `estimator_to_spec`, reference-data helpers, and a supported-estimator matrix in the README.
- Structure-backed inference on the API: executable uploads can be explained, predicted, and analysed without a remote URL.
- `client.health()`, `Page.auto_paging_iter()`, and `*.iterate()` helpers.
- Retries on 429 (and GET 502/503/504), honouring `Retry-After`.
- `auth.signup(..., turnstile_token=)` plus `verify_email` / `resend_verification`.
- Examples: `examples/upload_sklearn.py` and `examples/upload_sklearn.ipynb`.
- `PermissionDeniedError` (keeps `PermissionError` as an alias).

### Changed

- Version bump from 0.1.0. Default HTTP timeout is 120 seconds.
- `upload()` no longer PUTs `/structure` after `POST /v1/models` already stored it.
- Reference rows are capped at 5,000 with a warning. Large JSON specs warn at 2 MB and 16 MB.
- Non-executable uploads (importance summaries, approximate nets) warn that explain/predict need a remote URL.

### Packaging

- Extra `kaimei[models]` installs NumPy.
- CI runs Ruff, mypy, and Python 3.9–3.13.
