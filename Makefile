
RUN_ARGS += --user=$(shell id --user):$(shell id --group) --ulimit="core=0"

export DOCKER_BUILDKIT=1
export BETTER_EXCEPTIONS=1

help:
	@cat Makefile

update:
	uv sync --upgrade
	$(MAKE) test

format:
	-uv run pyfltr --exit-zero-even-if-formatted --commands=fast

test:
	uv run pyfltr --exit-zero-even-if-formatted

shell:
	uv run bash

.PHONY: help update format test shell
