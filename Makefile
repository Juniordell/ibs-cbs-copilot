.PHONY: migrate migrate-local migrate-neon test lint dev

migrate-local:
	@for f in migrations/*.sql; do \
		echo "→ $$f"; \
		docker compose exec -T postgres psql -U copilot -d copilot -f - < $$f; \
	done

migrate-neon:
	@if [ -z "$$NEON_URL" ]; then echo "Set NEON_URL first"; exit 1; fi
	@for f in migrations/*.sql; do \
		echo "→ $$f"; \
		psql "$$NEON_URL" -f $$f; \
	done

test:
	poetry run pytest -v

lint:
	poetry run ruff check .

dev:
	docker compose up -d

.DEFAULT_GOAL := help
help:
	@echo "make migrate-local   apply migrations to local Postgres"
	@echo "make migrate-neon    apply migrations to Neon (needs NEON_URL)"
	@echo "make test            run tests"
	@echo "make lint            run ruff"
	@echo "make dev             start docker compose"