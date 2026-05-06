.PHONY: help lint typecheck test test-cov format install install-dev clean

help:
	@echo "poker development commands:"
	@echo "  make install      - install in editable mode"
	@echo "  make install-dev  - install with dev dependencies"
	@echo "  make lint         - run ruff checks"
	@echo "  make format       - format code with ruff"
	@echo "  make typecheck    - run mypy strict type checking"
	@echo "  make test         - run pytest"
	@echo "  make test-cov     - run pytest with coverage report"
	@echo "  make clean        - remove build artifacts and caches"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/ tests/

test:
	pytest

test-cov:
	pytest --cov=src/poker --cov-report=html --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .coverage -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
