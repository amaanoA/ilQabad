.PHONY: install test lint format run clean

install:
	poetry install

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=src --cov-report=html

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

run:
	poetry run python -m src.presentation.cli.main

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf htmlcov .coverage
