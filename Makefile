FORMAT_TARGET = skqulacs tests

.PHONY: test
test:
	python -m pytest

.PHONY: format
format:
	black $(FORMAT_TARGET)

.PHONY: format_check
format_check:
	black --check --diff $(FORMAT_TARGET)