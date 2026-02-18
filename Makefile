.PHONY: install test lint format serve train-catboost train-prophet train-tft train-ensemble prepare-data clean smoke-test smoke-test-fast smoke-test-minimal generate-holidays backfill-epias help

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

clean: ## Remove build/cache artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

smoke-test: ## Smoke test all models
	uv run python scripts/smoke_test.py --config configs/smoke_test.yaml

smoke-test-fast: ## Smoke test (skip TFT)
	uv run python scripts/smoke_test.py --config configs/smoke_test.yaml --skip-tft

smoke-test-minimal: ## Smoke test (CatBoost only)
	uv run python scripts/smoke_test.py --config configs/smoke_test.yaml --skip-prophet --skip-tft

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
