from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import Project
from ..spec import TaskSpec

PIPELINE_TAG = {
    ("audio", "frame"): "voice-activity-detection",
    ("audio", "classify"): "audio-classification",
    ("text", "classify"): "text-classification",
    ("text", "frame"): "token-classification",
    ("image", "classify"): "image-classification",
    ("document", "classify"): "image-classification",
}
TASK_NAME = {
    "voice-activity-detection": "Voice Activity Detection",
    "audio-classification": "Audio Classification",
    "text-classification": "Text Classification",
    "token-classification": "Token Classification",
    "image-classification": "Image Classification",
}
NC_DATASETS = {"ashraq/esc50": "ESC-50 (CC BY-NC 3.0)"}


def fmt(x: Any, nd: int = 4) -> str:
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def lineage(p: Project, exp: str) -> list[dict]:
    chain, cur = [], exp
    while cur:
        f = p.exp_dir(cur) / "proposal.json"
        if not f.exists():
            chain.append({"exp": cur, "applied": []})
            break
        pr = json.loads(f.read_text())
        chain.append({"exp": cur, "parent": pr["parent"], "applied": pr["applied"]})
        cur = pr["parent"]
    return list(reversed(chain))


def suite_dataset(spec: TaskSpec, suite: str) -> tuple[str, str]:
    s = next(e for e in spec.evals if e.name == suite)
    if s.kind == "hf" and s.source:
        return s.source.dataset, s.source.dataset
    return f"{spec.name}-{suite}", f"byosm/{spec.name}-{suite}"


@dataclass
class CardFacts:
    p: Project
    repo_id: str
    license: str
    base: TaskSpec
    spec: TaskSpec
    manifest: dict
    meta: dict
    board: dict
    exp: str
    tag: str
    prim: str
    datasets: list[str]
    nc: list[str]
    lineage: list[dict]

    @property
    def results(self) -> dict:
        return self.board.get(self.exp, {})

    @property
    def encoders(self) -> list[str]:
        spec = self.spec
        if spec.modality == "document":
            backbones = (spec.document.text_backbone, spec.document.image_backbone)
        else:
            backbones = (spec.backbone,)
        return [b[3:] for b in backbones if b.startswith("hf:")]

    @property
    def jepa(self) -> bool:
        return self.spec.train.pretrain_steps > 0

    @property
    def repo_name(self) -> str:
        return self.repo_id.split("/")[-1]


def load_facts(name: str, repo_id: str, license: str) -> CardFacts:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    m = json.loads((p.root / "export" / "manifest.json").read_text())
    exp = m["experiment"]
    spec = TaskSpec.load(p.exp_dir(exp) / "spec.yaml")
    datasets = sorted({s.dataset for s in spec.data.hf_sources} | {e.source.dataset for e in base.evals if e.source})
    return CardFacts(
        p=p,
        repo_id=repo_id,
        license=license,
        base=base,
        spec=spec,
        manifest=m,
        meta=json.loads((p.exp_dir(exp) / "meta.json").read_text()),
        board=p.board(),
        exp=exp,
        tag=PIPELINE_TAG.get((spec.modality, spec.task_type), "other"),
        prim=base.primary_suite(),
        datasets=datasets,
        nc=[NC_DATASETS[d] for d in datasets if d in NC_DATASETS],
        lineage=lineage(p, exp),
    )
