from __future__ import annotations

import hashlib
from pathlib import Path

from ..deploy import DOCKER_DIR, build_image, docker

TRAIN_IMAGE = "byosm-train"
TRAIN_DOCKERFILE = DOCKER_DIR / "train.Dockerfile"
PYPROJECT = DOCKER_DIR.parent / "pyproject.toml"
TORCH_INDEX = {"cpu": "https://download.pytorch.org/whl/cpu", "gpu": "https://pypi.org/simple"}
TAG_DIGITS = 8


def training_image(gpu: bool) -> dict[str, str]:
    from ..app.snapshot import frozen

    code, flavour = frozen(), "gpu" if gpu else "cpu"
    digest = hashlib.sha1(f"{code.name}-{flavour}".encode())
    for source in (PYPROJECT, TRAIN_DOCKERFILE):
        data, target = source.read_bytes(), code / source.name
        digest.update(data)
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
    tag = f"{TRAIN_IMAGE}:{code.name}-{digest.hexdigest()[:TAG_DIGITS]}"
    return {"image": tag, "context": str(code), "torch": TORCH_INDEX[flavour]}


def ensure_image(image: str, context: str, torch: str) -> None:
    if docker("image", "inspect", image).returncode == 0:
        print(f"  training image {image}", flush=True)
        return
    build_image(Path(context) / TRAIN_DOCKERFILE.name, Path(context), image, print, {"TORCH_INDEX": torch})
