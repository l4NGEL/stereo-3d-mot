# Linux build/runtime environment for stereo_3d_mot.
# Develop on the host, build inside this image so the toolchain matches CI and
# the deployment target.

# ---- deps: toolchain + libraries (cached layer) --------------------------
FROM ubuntu:22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        pkg-config \
        libopencv-dev \
        libeigen3-dev \
        libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- build: compile the project (a green image == the tree builds) -------
FROM deps AS build
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build build --parallel

CMD ["/bin/bash"]
