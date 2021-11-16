PYSEN := poetry run pysen
PYTEST := poetry run pytest
SPHINX_APIDOC := poetry run sphinx-apidoc
PROJECT_DIR := project_name
PORT := 8000

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: test
test:
	$(PYTEST) -v

tests/%.py: FORCE
	$(PYTEST) $@

.PHONY: lint
lint:
	$(PYSEN) run lint

.PHONY: format
format:
	$(PYSEN) run format

.PHONY: check
check: format lint

.PHONY: api
api:
	$(SPHINX_APIDOC) -f -e -o doc/source $(PROJECT_DIR)

.PHONY: doc
html: api
	poetry run $(MAKE) -C doc html

.PHONY: serve
serve: html
	poetry run python -m http.server --directory doc/build/html $(PORT)