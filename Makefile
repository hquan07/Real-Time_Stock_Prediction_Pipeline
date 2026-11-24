PYTHON ?= python3
COMPOSE_ENV ?= .env.example

.PHONY: bootstrap compose-config frontend-install lint test-unit build smoke-test

bootstrap: frontend-install
	$(PYTHON) -m pip install -r requirements-base.txt -r requirements-dev.txt

compose-config:
	docker compose --env-file $(COMPOSE_ENV) config --quiet

frontend-install:
	npm --prefix apps/web ci

lint: compose-config
	./scripts/ci/check_secrets.sh
	$(PYTHON) -m compileall -q src services scripts tests
	npm --prefix apps/web run lint

test-unit:
	$(PYTHON) -m pytest -c tests/pytest.ini tests/unit

build:
	npm --prefix apps/web run build
	docker compose --env-file $(COMPOSE_ENV) build dashboard-backend dashboard-frontend

smoke-test:
	./scripts/ci/smoke_dashboard.sh
