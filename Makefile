FORMAT_TARGET = skqulacs tests

.PHONY: install
install:
	pip install -e .

.PHONY: test
test:
	python -m pytest -v

.PHONY: format
format:
	python -m black $(FORMAT_TARGET)

.PHONY: format_check
format_check:
	python -m black --check --diff $(FORMAT_TARGET)