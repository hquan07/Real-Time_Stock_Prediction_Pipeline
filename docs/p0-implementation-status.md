# P0 implementation status

Updated: 2026-11-24
Branch: `upr/p0-foundation`

## Completed foundation slices

- Repository runtime hygiene: `.env`, IDE metadata, virtual environments, logs,
  notebook checkpoints and Python caches are no longer tracked; local copies
  were preserved.
- Canonical dashboard paths: React/Vite moved to `apps/web`, FastAPI moved to
  `services/api`, and legacy Dash is frozen under `src/dashboard`.
- Runtime trust fixes: Overview is the default, mock architecture is removed
  from navigation, alert seeds and synthetic uncertainty are removed, and API
  plus WebSocket URLs use same-origin configuration.
- Shared settings: database, Kafka, Redis, schema path and CORS configuration
  now come from environment-backed settings; committed password fallbacks and
  absolute project paths were removed.
- Dashboard runtime: non-root images, Nginx reverse proxy, service healthchecks,
  single-broker Kafka settings and port `5173` avoid the existing host port 80.
- Quality baseline: Make targets, local pre-commit hooks, committed-secret
  check, GitHub Actions gates, targeted Python tests, frontend lint/build and
  dashboard smoke checks are present.

## Verified evidence

- `npm run lint` passed.
- `npm run build` passed; the bundle-size warning remains open.
- Targeted Python 3.12 tests passed: 14 tests.
- `docker compose config --quiet` passed with `.env.example`.
- PostgreSQL, Kafka, API and frontend healthchecks passed; `/api/health` passed
  through the frontend reverse proxy.
- `git ls-files` reports no tracked `.env`, virtualenv, cache, log or notebook
  checkpoint paths.

## Remaining P0 work

- Rotate all exposed external credentials and approve coordinated Git-history
  rewrite before any remote push; current-tree cleanup does not revoke secrets.
- Move pipeline business logic into `packages/pipeline/src/stock_pipeline` and
  reduce scripts/DAGs to thin adapters.
- Convert the role-bound SQL dump into versioned, role-neutral migrations and
  separate schema from demo seed data.
- Consolidate Python/Airflow dependency sets with a reproducible constraints or
  lock strategy.
- Complete UI loading, empty, error, retry, freshness, source and version states.
- Add healthchecks/network boundaries for the non-dashboard services and run the
  full clean-environment test matrix.
