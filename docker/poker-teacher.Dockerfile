FROM rust:1.99-trixie@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137 AS build
ARG REPO=https://github.com/conorarmstrong/pluribus.git
ARG COMMIT
WORKDIR /src
RUN git clone ${REPO} . && git checkout ${COMMIT}
COPY poker-teacher.patch /tmp/teacher.patch
RUN git apply /tmp/teacher.patch && cargo test --release benchmark && cargo test --release checkpoint && cargo build --release

FROM debian:stable-slim@sha256:eb593cf2c358cacef45ca0a424bbc7d30cfa3466265fc2662b9466a0ca6ba1c5
RUN useradd --create-home --uid 10001 teacher
COPY --from=build /src/target/release/pluribus /usr/local/bin/pluribus
USER teacher
WORKDIR /work
ENTRYPOINT ["pluribus"]
