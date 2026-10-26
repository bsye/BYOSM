from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import HF_TOKEN_KEY, load_config, secret
from .project import WEIGHTS, Project

REMOTE_KEY = "artifact_repo"
REPO_TYPE = "model"
EXPORT_DIR = "export"


def remote(ws: Path) -> str | None:
    return load_config(ws).get(REMOTE_KEY)


def _base(p: Project) -> Path:
    return p.root.parent.parent


def _prefix(p: Project) -> str:
    return p.root.relative_to(_base(p)).as_posix()


def _patterns(p: Project, exp: str | None) -> list[str]:
    prefix = _prefix(p)
    if exp:
        return [f"{prefix}/experiments/{exp}/*"]
    return [f"{prefix}/experiments/*/*", f"{prefix}/{EXPORT_DIR}/**"]


def push(p: Project, exp: str | None = None, log: Callable[[str], None] = print) -> str:
    from huggingface_hub import HfApi

    repo = remote(p.ws)
    if not repo:
        raise ValueError(f"set {REMOTE_KEY} in byosm.yaml to a Hugging Face repo id, e.g. you/byosm-store")
    api = HfApi(token=secret(HF_TOKEN_KEY, p.ws))
    api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise RuntimeError(f"{repo} is public; the artifact store must be private")
    patterns = _patterns(p, exp)
    api.upload_folder(repo_id=repo, repo_type=REPO_TYPE, folder_path=_base(p), allow_patterns=patterns)
    log(f"pushed {', '.join(patterns)} to {repo}")
    return repo


def pull(p: Project, exp: str | None = None, log: Callable[[str], None] = print) -> Path:
    from huggingface_hub import snapshot_download

    repo = remote(p.ws)
    if not repo:
        raise ValueError(f"set {REMOTE_KEY} in byosm.yaml to pull artifacts")
    patterns = _patterns(p, exp)
    snapshot_download(
        repo, repo_type=REPO_TYPE, local_dir=_base(p), allow_patterns=patterns, token=secret(HF_TOKEN_KEY, p.ws)
    )
    log(f"pulled {', '.join(patterns)} from {repo}")
    return p.root


def weights(p: Project, exp: str) -> Path:
    path = p.exp_dir(exp) / WEIGHTS
    if not path.exists() and remote(p.ws):
        pull(p, exp, log=lambda _: None)
    return path
