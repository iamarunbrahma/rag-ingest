.PHONY: lint format

lint:
	ruff check . --fix

format:
	black .