TARGET_DIR := skqulacs tests
PIP_INSTALL := pip install
PYTEST := python -m pytest -v
BLACK := python -m black
FLAKE8 := python -m flake8
FLAKE8_IGNORE := --ignore=E501

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
	$(BLACK) $(TARGET_DIR)

.PHONY: format_check
format_check:
	$(BLACK) --check --diff $(TARGET_DIR)

.PHONY: lint
lint:
	$(FLAKE8) $(FLAKE8_IGNORE) $(TARGET_DIR)