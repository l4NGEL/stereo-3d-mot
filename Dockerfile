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
        curl \
        git \
        ca-certificates \
        pkg-config \
        libopencv-dev \
        libeigen3-dev \
        libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime (CPU) -- prebuilt release, used by the optional OnnxDetector.
ARG ORT_VERSION=1.19.2
RUN curl -fsSL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib

WORKDIR /workspace

# ---- build: compile the project (a green image == the tree builds) -------
FROM deps AS build
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build build --parallel

CMD ["/bin/bash"]
