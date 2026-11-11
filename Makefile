# MLBreak — convenience targets. Uses a local virtualenv in .venv.
PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin
# PYTHONPATH=src keeps commands working even if a synced folder hides the
# editable-install .pth file (a known macOS issue with Python >= 3.12).
RUN    := PYTHONPATH=src $(BIN)/python
PORT   ?= 8000

.PHONY: install ui run dev test verify demo docker clean
.DEFAULT_GOAL := run

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

install: $(BIN)/python
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

ui:
	cd frontend && npm ci --no-audit --no-fund && npm run build

run: ## serve API + built UI on http://127.0.0.1:$(PORT)
	$(RUN) -m mlbreak serve --port $(PORT)

dev: ## Vite dev server (proxies /api to :8000; run `make run` in another shell)
	cd frontend && npm run dev

test:
	$(RUN) -m pytest

verify: ## security regression gate (non-zero exit on failure)
	$(RUN) -m mlbreak verify

demo:
	$(RUN) -m mlbreak demo

docker:
	docker compose up --build

clean:
	rm -rf var frontend/dist .pytest_cache
