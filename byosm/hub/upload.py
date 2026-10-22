from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from ..config import PUBLISH_LICENSE
from ..evals.dumps import _dumps
from ..export.bundle import is_exported
from ..model_refs import HUB_DIR, HUB_SEPARATOR
from ..project import Project
from .card import model_card
from .facts import lineage


def _stage(p: Project, repo_id: str, license: str, experiment: str) -> Path:
    export_dir = p.root / "export"
    stage = p.root / HUB_DIR / repo_id.replace("/", HUB_SEPARATOR)
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(export_dir, stage, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    p.exp_spec(experiment).save(stage / "spec.yaml")
    (stage / "board.json").write_text(_dumps(p.board()))
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
    if not is_exported(p.root / "export"):
        raise RuntimeError(f"export first: byosm export {name}")
    experiment = json.loads(manifest.read_text())["experiment"]
    if experiment != p.promoted:
        raise RuntimeError(f"the export holds {experiment} but {p.promoted} is promoted: byosm export {name} first")
    stage = _stage(p, repo_id, license, experiment)
    log(f"  staged {len(list(stage.iterdir()))} files in {stage}")
    if dry_run:
        log("  dry run: nothing uploaded")
        return stage
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    if private and not api.model_info(repo_id).private:
        raise RuntimeError(f"{repo_id} already exists and is public; pass --public to publish to it")
    api.upload_folder(folder_path=str(stage), repo_id=repo_id, commit_message=f"BYOSM publish {name}/{experiment}")
    log(f"  published https://huggingface.co/{repo_id} ({'private' if private else 'public'})")
    return stage
