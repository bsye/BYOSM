ARG TEACHER
ARG RIVAL
FROM ${TEACHER} AS teacher
FROM ${RIVAL} AS rival

FROM debian:stable-slim@sha256:eb593cf2c358cacef45ca0a424bbc7d30cfa3466265fc2662b9466a0ca6ba1c5
ARG NUMPY
ARG ONNXRUNTIME
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && pip install --no-cache-dir --break-system-packages numpy==${NUMPY} onnxruntime==${ONNXRUNTIME} \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 player
COPY --from=teacher /usr/local/bin/pluribus /usr/local/bin/poker-engine
COPY --from=rival /usr/local/bin/bench /usr/local/bin/bench
USER player
WORKDIR /work
