PYTEST := poetry run pytest
FORMATTER := poetry run black
LINTER := poetry run flake8
IMPORT_SORTER := poetry run isort
TYPE_CHECKER := poetry run mypy
SPHINX_APIDOC := poetry run sphinx-apidoc

PROJECT_DIR := skqulacs
TEST_DIR := tests
BENCHMARK_DIR := benchmarks
CHECK_DIR := $(PROJECT_DIR) $(TEST_DIR) $(BENCHMARK_DIR)

COVERAGE_OPT := --cov skqulacs --cov-branch
BENCHMARK_OPT := --benchmark-autosave -v
PORT := 8000

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: check
check: format lint

.PHONY: ci
ci: format_check lint

.PHONY: test
test:
	$(PYTEST) -v $(TEST_DIR)

tests/%.py: FORCE
	$(PYTEST) $@

.PHONY: format
format:
	$(FORMATTER) $(CHECK_DIR)
	$(IMPORT_SORTER) $(CHECK_DIR)

.PHONY: format_check
format_check:
	$(FORMATTER) $(CHECK_DIR) --check --diff
	$(IMPORT_SORTER) $(CHECK_DIR) --check --diff

.PHONY: cov
cov:
	$(PYTEST) $(COVERAGE_OPT) --cov-report html $(TEST_DIR)

.PHONY: cov_ci
cov_ci:
	$(PYTEST) $(COVERAGE_OPT) --cov-report xml $(TEST_DIR)

.PHONY: serve_cov
serve_cov: cov
	poetry run python -m http.server --directory htmlcov $(PORT)

.PHONY: lint
lint:
	$(LINTER) $(CHECK_DIR)

.PHONY: type
type:
	$(TYPE_CHECKER) $(CHECK_DIR)

.PHONY: benchmark
benchmark:
	$(PYTEST) $(BENCHMARK_DIR) $(BENCHMARK_OPT)

benchmarks/%.py: FORCE
	$(PYTEST) $@ $(BENCHMARK_OPT)

.PHONY: serve
serve: html
	poetry run python -m http.server --directory doc/build/html $(PORT)

.PHONY: doc
html: api
	poetry run $(MAKE) -C doc html

.PHONY: api
api:
	$(SPHINX_APIDOC) -f -e -o doc/source $(PROJECT_DIR)
