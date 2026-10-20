from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from ..config import PUBLISH_LICENSE, Project
from .card import model_card
from .facts import lineage


def _stage(p: Project, repo_id: str, license: str, experiment: str) -> Path:
    export_dir = p.root / "export"
    stage = p.root / "hub" / repo_id.replace("/", "__")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(export_dir, stage, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    shutil.copy(p.exp_dir(experiment) / "spec.yaml", stage / "spec.yaml")
    shutil.copy(p.board_path, stage / "board.json")
    (stage / "lineage.json").write_text(json.dumps(lineage(p, experiment), indent=2))
    (stage / "README.md").write_text(model_card(p.name, repo_id, license))
    return stage


def publish(
    name: str,
    repo_id: str,
    license: str = PUBLISH_LICENSE,
    private: bool = True,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    p = Project(name)
    manifest = p.root / "export" / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"export first: byosm export {name}")
    experiment = json.loads(manifest.read_text())["experiment"]
    stage = _stage(p, repo_id, license, experiment)
    log(f"  staged {len(list(stage.iterdir()))} files in {stage}")
    if dry_run:
        log("  dry run: nothing uploaded")
        return stage
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=repo_id, commit_message=f"BYOSM publish {name}/{experiment}")
    log(f"  published https://huggingface.co/{repo_id} ({'private' if private else 'public'})")
    return stage
