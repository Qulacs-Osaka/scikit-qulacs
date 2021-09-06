TARGET_DIR := skqulacs tests
PIP_INSTALL := pip install
PYTEST := python -m pytest -v
FORMATTER := python -m black
FLAKE8 := python -m flake8
FLAKE8_OPTS := 
AUTOPEP8 := python -m autopep8
AUTOPEP8_OPTS := -r --exit-code --verbose

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
	$(AUTOPEP8) $(AUTOPEP8_OPTS) --in-place $(TARGET_DIR)

.PHONY: lint_check
lint_check:
	$(FLAKE8) $(TARGET_DIR)

.PHONY: fix
fix: format lint

.PHONY: check
check: format_check lint_check
