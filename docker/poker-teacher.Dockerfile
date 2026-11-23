FROM rust:1.99-trixie AS build
ARG REPO=https://github.com/conorarmstrong/pluribus.git
ARG COMMIT
WORKDIR /src
RUN git clone ${REPO} . && git checkout ${COMMIT}
COPY poker-teacher-label.patch /tmp/label.patch
RUN git apply /tmp/label.patch && cargo test --release benchmark && cargo build --release

FROM debian:stable-slim
RUN useradd --create-home --uid 10001 teacher
COPY --from=build /src/target/release/pluribus /usr/local/bin/pluribus
USER teacher
WORKDIR /work
ENTRYPOINT ["pluribus"]
