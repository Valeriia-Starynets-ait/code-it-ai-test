.PHONY: check lint typecheck test

# The quality gate the agent runs before opening a pull request.
check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

test:
	uv run python -m pytest
