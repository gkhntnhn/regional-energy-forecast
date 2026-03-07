.PHONY: install test lint format serve train-catboost train-prophet train-tft train-ensemble prepare-data clean generate-holidays backfill-epias db-up db-down db-migrate db-revision db-downgrade fetch-weather-actuals db-backup promote-model cleanup-old-data cleanup-dry-run help

install: ## Install dependencies
	uv sync --all-extras

test: ## Run unit tests
	uv run pytest -x --tb=short

lint: ## Run linters (ruff + mypy)
	uv run ruff check src/ tests/
	uv run mypy src/

format: ## Auto-format code
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

serve: ## Start FastAPI dev server
	uv run uvicorn energy_forecast.serving.app:app --host 0.0.0.0 --port 8000 --reload

train-catboost: ## Train CatBoost model
	uv run python -m energy_forecast.training.run --model catboost

train-prophet: ## Train Prophet model
	uv run python -m energy_forecast.training.run --model prophet

train-tft: ## Train TFT model
	uv run python -m energy_forecast.training.run --model tft

train-ensemble: ## Train ensemble (all models + weight optimization)
	uv run python -m energy_forecast.training.run --model ensemble

prepare-data: ## Prepare dataset (Excel -> feature parquets)
	uv run python scripts/prepare_dataset.py

generate-holidays: ## Generate Turkish holidays parquet
	uv run python scripts/generate_holidays.py

backfill-epias: ## Backfill EPIAS market data cache
	uv run python scripts/backfill_epias.py

db-up: ## Start PostgreSQL (Docker Compose)
	docker compose up -d db

db-down: ## Stop PostgreSQL
	docker compose down

db-migrate: ## Run Alembic migrations (upgrade head)
	uv run alembic upgrade head

db-revision: ## Create new Alembic revision (MSG="description")
	uv run alembic revision --autogenerate -m "$(MSG)"

db-downgrade: ## Downgrade one migration
	uv run alembic downgrade -1

fetch-weather-actuals: ## Fetch weather actuals for T-2 day
	uv run python -m energy_forecast.jobs.weather_actuals

db-backup: ## Backup database to gzipped SQL dump
	uv run python scripts/backup_db.py

promote-model: ## Promote best model run to final_models/ (MODEL=catboost)
	uv run python scripts/promote_model.py --model $(MODEL)

cleanup-old-data: ## Apply retention policy (90 days default)
	uv run python scripts/cleanup_jobs.py --days 90

cleanup-dry-run: ## Show what would be deleted (dry run)
	uv run python scripts/cleanup_jobs.py --days 90 --dry-run

clean: ## Remove build/cache artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
