FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libomp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSELDON_ENABLE_OPENMP=ON \
    -DSELDON_ENABLE_REST_SERVICE=ON \
    -DSELDON_ENABLE_OPENCL=OFF \
    -DSELDON_ENABLE_CUDA=OFF \
    && cmake --build build -j"$(nproc)" --target seldon

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/seldon /usr/local/bin/seldon

EXPOSE 8090
EXPOSE 8091

ENTRYPOINT ["/usr/local/bin/seldon"]