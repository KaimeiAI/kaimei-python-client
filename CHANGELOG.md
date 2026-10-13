# Changelog

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
