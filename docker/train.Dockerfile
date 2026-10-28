FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/opt/byosm
RUN pip install --index-url ${TORCH_INDEX} torch torchaudio torchvision
WORKDIR /opt/build
COPY pyproject.toml ./
RUN mkdir byosm && touch byosm/__init__.py && pip install ".[app]" && pip uninstall --yes byosm && rm -rf /opt/build
COPY byosm/ /opt/byosm/byosm/
WORKDIR /
