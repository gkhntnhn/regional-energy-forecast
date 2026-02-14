.PHONY: install test lint format serve train-catboost train-prophet train-tft train-ensemble prepare-data clean smoke-test smoke-test-fast smoke-test-minimal

install:
	uv sync --all-extras

test:
	uv run pytest -x --tb=short

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

serve:
	uv run uvicorn energy_forecast.serving.app:app --host 0.0.0.0 --port 8000 --reload

train-catboost:
	uv run python -m energy_forecast.training.run --model catboost

train-prophet:
	uv run python -m energy_forecast.training.run --model prophet

train-tft:
	uv run python -m energy_forecast.training.run --model tft

train-ensemble:
	uv run python -m energy_forecast.training.run --model ensemble

prepare-data:
	uv run python scripts/prepare_dataset.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

smoke-test:
	uv run python scripts/smoke_test.py --config configs/smoke_test.yaml

smoke-test-fast:
	uv run python scripts/smoke_test.py --config configs/smoke_test.yaml --skip-tft

smoke-test-minimal:
	uv run python scripts/smoke_test.py --config configs/smoke_test.yaml --skip-prophet --skip-tft
