FROM node:20-slim AS ui
WORKDIR /src/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BYOSM_CACHE=/cache/byosm HF_HOME=/cache/hf
RUN pip install --index-url ${TORCH_INDEX} torch torchaudio torchvision
WORKDIR /opt/byosm
COPY pyproject.toml LICENSE ./
COPY byosm/ byosm/
COPY --from=ui /src/byosm/app/static/ byosm/app/static/
RUN pip install ".[app]" && rm -rf /opt/byosm
RUN useradd --create-home --uid 1000 byosm && mkdir -p /workspace /cache && chown byosm:byosm /workspace /cache
USER byosm
WORKDIR /workspace
VOLUME ["/workspace", "/cache"]
EXPOSE 8765
CMD ["byosm", "app", "--host", "0.0.0.0", "--port", "8765", "--no-open"]
