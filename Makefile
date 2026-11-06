.PHONY: check lint test

# The quality gate the agent runs before opening a pull request.
check: lint test

lint:
	uv run --with ruff ruff check .

test:
	uv run --with pytest python -m pytest
