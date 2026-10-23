from __future__ import annotations

import json
import shutil
from pathlib import Path

FOLDER = "encoders"
EMBEDDED = {
    "convaiinnovations/laya": {"config": "encoder/config.json", "tokenizer": "tokenizer", "prefix": "encoder."},
}
WEIGHTS = "model.safetensors"
ENCODER_ARCHITECTURE = {"modernbert": "ModernBertModel"}


def _folder(repo: str) -> Path:
    from .config import cache_root

    return cache_root() / FOLDER / repo.replace("/", "--")


def _extract(repo: str, how: dict[str, str], out: Path) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open
    from safetensors.torch import save_file

    tmp = out.with_suffix(".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    config = json.loads(Path(hf_hub_download(repo, how["config"])).read_text())
    config["architectures"] = [ENCODER_ARCHITECTURE.get(config.get("model_type", ""), config["architectures"][0])]
    (tmp / "config.json").write_text(json.dumps(config, indent=1))
    for name in list_repo_files(repo):
        if name.startswith(how["tokenizer"] + "/"):
            shutil.copy(hf_hub_download(repo, name), tmp / name.rpartition("/")[2])
    with safe_open(hf_hub_download(repo, WEIGHTS), "pt") as f:
        names = f.keys()
        tensors = {k.removeprefix(how["prefix"]): f.get_tensor(k) for k in names if k.startswith(how["prefix"])}
    save_file(tensors, str(tmp / WEIGHTS))
    tmp.rename(out)


def resolve(repo: str) -> str:
    how = EMBEDDED.get(repo)
    if how is None:
        return repo
    out = _folder(repo)
    if not out.exists():
        _extract(repo, how, out)
    return str(out)
