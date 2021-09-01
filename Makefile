TARGET_DIR := skqulacs tests
PIP_INSTALL := pip install
PYTEST := python -m pytest -v
FORMATTER := python -m black
LINTER := python -m autopep8
LINTER_OPTS := -r --exit-code

.PHONY: install
install:
	$(PIP_INSTALL) -e .

.PHONY: test
test:
	$(PYTEST)

tests/%.py: FORCE
	$(PYTEST) $@

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: format
format:
	$(FORMATTER) $(TARGET_DIR)

.PHONY: format_check
format_check:
	$(FORMATTER) --check --diff $(TARGET_DIR)

.PHONY: lint
lint:
	$(LINTER) $(LINTER_OPTS) --in-place $(TARGET_DIR)

.PHONY: lint_check
lint_check:
	$(LINTER) $(LINTER_OPTS) --diff $(TARGET_DIR)

.PHONY: fix
fix: format lint

.PHONY: check
check: format_check lint_check
