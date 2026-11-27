# P0 implementation status

Updated: 2026-11-26
Branch: `upr/p0-foundation`

## Outcome

All repository-controlled P0 foundation work is implemented. The codebase now
has one canonical Python package, versioned migrations, explicit data and model
correctness gates, a truthful dashboard contract, pinned runtime boundaries and
CI coverage for the release-critical paths.

The P0 branch is ready for remote review. This does not approve a pilot release;
later-phase controls and remote CI still require review.

## Completed scope

- Repository hygiene: local environments, caches, logs, checkpoints and model
  binaries are ignored and removed from Git tracking without deleting local
  runtime copies.
- Runtime architecture: application logic lives in
  `packages/pipeline/src/stock_pipeline`; scripts and Airflow DAGs are adapters;
  legacy namespaces are compatibility-only.
- Configuration: database, Kafka, Redis, CORS, schema and artifact locations are
  environment-backed and required secrets fail closed.
- Data contract: versioned Avro v2 events carry deterministic event IDs, UTC
  millisecond timestamps, source/interval/timezone metadata and OHLCV/session
  validation.
- Storage: role-neutral, checksum-verified migrations support advisory locking,
  rollback and idempotent reapply; demo seed data is separate.
- Streaming: the Spark consumer uses pinned Kafka/Avro/PostgreSQL connectors,
  persistent checkpoints and event-ID upsert semantics. Its healthcheck requires
  both the running job and initialized checkpoint metadata.
- ML correctness: features and targets are ticker-isolated; chronological split
  and train-only scaling prevent leakage. Random forest promotion requires
  beating the zero-return baseline on holdout and expanding walk-forward tests.
  LSTM remains ineligible until equivalent validation exists.
- Signals: P0 emits research-only HOLD output with measured RSI and explicit
  unavailable confidence instead of fabricated trading certainty.
- API and dashboard: responses use `{data,state,meta}` with provenance,
  freshness, timezone, market status and model version. Loading, empty, error,
  retry and boundary-error states are visible; portfolio values are calculated
  server-side; research disclaimers are persistent.
- Delivery: Python dependency groups and constraints, non-root images,
  loopback-only host bindings, environment templates, a pilot Compose overlay,
  pre-commit hooks and GitHub Actions quality gates are present.

## Verification evidence

Evidence captured on 2026-11-25:

- Python 3.12 non-Spark suite: **88 passed, 4 skipped**.
- Pinned Spark-image suite as non-root with read-only source: **12 passed**.
- Frontend lint and production build passed. Vite still reports a non-blocking
  approximately 5.38 MB bundle-size warning.
- Python compilation, structured YAML/JSON validation, committed-secret scan and
  `docker compose --env-file .env.example config --quiet` passed.
- `pip-audit` reported no known vulnerabilities for the canonical pipeline and
  API requirement sets after upgrading the affected pins.
- Trivy reported zero fixable HIGH/CRITICAL findings for both dashboard images.
  The API image upgrades OS packages during build and removes `pip`/`setuptools`
  after dependency installation, so build tooling and its vendored SBOM are not
  present in the runtime image; CI asserts this before scanning.
- Migration drill passed on an isolated database: apply `0001`/`0002`, roll back
  `0002`, then reapply. The temporary database was removed after verification.
- Live dashboard stack healthchecks and reverse-proxy smoke test passed;
  no-data endpoints returned explicit empty envelopes instead of mock values.
- Spark restart recovery resumed from persisted checkpoint batch/offset metadata.
- Core local services were healthy after the drill: PostgreSQL, Kafka, Spark
  consumer, dashboard API and dashboard frontend.

## Credential and history decision

On 2026-11-26, the owner reported that the old Finnhub key and CloudVault token
were revoked at their providers. This is owner attestation; provider-side
revocation was not independently verified by the repository checks.

The ignored local `.env` now has a new Finnhub key and no CloudVault token.
PostgreSQL password, Airflow connection/admin password and Jupyter token were
rotated locally. The old PostgreSQL password was rejected over the application
network, the new password was accepted, API/Spark restarted with the new value,
and the dashboard smoke test passed. Airflow user metadata was absent and
Jupyter was not running during rotation, so no active user/session for those
services was available to verify.

The owner chose to retain existing Git history after revocation. The old `.env`
remains in public historical commits, while the current tree no longer tracks
it. History removal would require a separate coordinated rewrite with
collaborators and potentially GitHub support. See
[`preflight/secret-remediation-runbook.md`](preflight/secret-remediation-runbook.md).

## Post-P0 boundary

Authentication/authorization, full replay and fault-injection coverage,
backtesting/paper-risk controls, production observability/SLOs, backup/restore
certification and pilot go/no-go remain later-phase work. Their absence does not
reopen the P0 repository foundation, but it prevents a production-readiness
claim.
