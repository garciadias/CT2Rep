.PHONY: help build test deps run clean lint shell

# load ariables from .env file
ifneq ("$(wildcard .env)","")
include .env
export $(shell sed 's/=.*//' .env)
endif

# Build Docker container
build:
	@echo "Building Docker container..."
	$(DOCKER_COMMAND) build 

# Run tests
test:
	@echo "Running tests..."
	$(DOCKER_COMMAND) run --rm \
	--name $(CONTAINER_NAME)-test \
	$(IMAGE_NAME) uv run pytest $(TEST_DIR) -v

# Open a shell in the container
shell:
	@echo "Opening shell in container..."
	$(DOCKER_COMMAND) run --rm -it --name $(CONTAINER_NAME)-shell $(IMAGE_NAME) /bin/bash
