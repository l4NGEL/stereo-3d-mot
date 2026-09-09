# Convenience wrapper around the Docker workflow. Every target runs inside the
# stereo-3d-mot:dev image with the working tree bind-mounted at /workspace, so
# ./build and ./out appear on the host.

IMAGE      ?= stereo-3d-mot:dev
BUILD_TYPE ?= RelWithDebInfo
DOCKER_RUN  = docker run --rm -v "$(CURDIR)":/workspace -w /workspace $(IMAGE)

.PHONY: help image configure build test demo bench shell fmt fmt-check clean

help:
	@echo "make image      build the Linux dev image"
	@echo "make build      configure + compile into ./build"
	@echo "make test       run the unit tests (ctest)"
	@echo "make demo       run the depth demo on the synthetic scene -> ./out"
	@echo "make bench      run the depth benchmark on the synthetic scene"
	@echo "make shell      interactive shell in the dev image"
	@echo "make fmt        reformat sources with clang-format"
	@echo "make clean      remove ./build and ./out"

image:
	docker build -t $(IMAGE) .

configure: image
	$(DOCKER_RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(DOCKER_RUN) cmake --build build --parallel

test: build
	$(DOCKER_RUN) ctest --test-dir build --output-on-failure

demo: build
	$(DOCKER_RUN) ./build/apps/stereo_depth_demo --source synthetic --out out --cloud

bench: build
	$(DOCKER_RUN) ./build/apps/benchmark_depth --source synthetic

shell: image
	docker run --rm -it -v "$(CURDIR)":/workspace -w /workspace $(IMAGE) bash

fmt: image
	$(DOCKER_RUN) bash -c "find include src apps tests -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i"

fmt-check: image
	$(DOCKER_RUN) bash -c "find include src apps tests -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format --dry-run --Werror"

clean:
	rm -rf build out
