FORMAT_TARGET = skqulacs tests

.PHONY: test
test:
	python -m pytest

.PHONY: format
format:
	python -m black $(FORMAT_TARGET)

.PHONY: format_check
format_check:
	python -m black --check --diff $(FORMAT_TARGET)