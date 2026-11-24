FROM rust:1.99-trixie@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137 AS build
ARG REPO
ARG COMMIT
WORKDIR /src
RUN git clone ${REPO} . && git checkout ${COMMIT}
RUN cargo build --release --locked -p bench -p pluribus -p sharingan-serve

FROM debian:stable-slim@sha256:eb593cf2c358cacef45ca0a424bbc7d30cfa3466265fc2662b9466a0ca6ba1c5
RUN useradd --create-home --uid 10001 player
COPY --from=build /src/target/release/bench /src/target/release/pluribus /src/target/release/sharingan-serve /usr/local/bin/
USER player
WORKDIR /work
