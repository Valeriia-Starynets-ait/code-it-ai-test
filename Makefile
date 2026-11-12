.PHONY: check lint format typecheck test

# The quality gate the agent runs before opening a pull request.
check: lint format typecheck test

lint:
	uv run --with ruff ruff check .

format:
	uv run --with ruff ruff format --check .

typecheck:
	uv run --extra dev --with pydantic --with pydantic-settings mypy src

test:
	uv run --extra dev python -m pytest
