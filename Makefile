.PHONY: test
test:
	python -m pytest

.PHONY: format
format:
	black skqulacs