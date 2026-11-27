# Environment deployment policy

- Local development uses the root `docker-compose.yml` and `.env.example`.
- Test, staging and pilot use separate credentials and databases; example
  values live in `config/environments/`.
- Pilot uses `pilot.yml`, published images tagged with the full Git SHA, and no
  source-code bind mounts. Real secrets must be injected by the deployment
  platform and must never be committed.
