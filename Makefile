.PHONY: lint format secrets-init

lint:
	ruff check . --fix

format:
	black .

secrets-init:
	detect-secrets scan --baseline .secrets.baseline