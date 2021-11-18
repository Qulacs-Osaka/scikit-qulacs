PYTEST := poetry run pytest
FORMATTER := poetry run black
LINTER := poetry run flake8
IMPORT_SORTER := poetry run isort
TYPE_CHECKER := poetry run mypy
TARGET_DIR := skqulacs tests
SPHINX_APIDOC := poetry run sphinx-apidoc
PORT := 8000

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: test
test:
	$(PYTEST) -v

tests/%.py: FORCE
	$(PYTEST) $@

.PHONY: check
check:
	$(FORMATTER) $(TARGET_DIR) --check --diff
	$(LINTER) $(TARGET_DIR)
	$(IMPORT_SORTER) $(TARGET_DIR) --check --diff

.PHONY: fix
fix:
	$(FORMATTER) $(TARGET_DIR)
	$(IMPORT_SORTER) $(TARGET_DIR)

.PHONY: type
type:
	$(TYPE_CHECKER) skqulacs

.PHONY: api
api:
	$(SPHINX_APIDOC) -f -e -o doc/source $(PROJECT_DIR)

.PHONY: doc
html: api
	poetry run $(MAKE) -C doc html

.PHONY: serve
serve: html
	poetry run python -m http.server --directory doc/build/html $(PORT)