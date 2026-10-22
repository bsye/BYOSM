from __future__ import annotations

import json
from collections.abc import Callable

from ..project import Project
from ..rt import tag_parts
from ..spec import DataPlan, EvalSuite, HFSource, Label, Objective, TaskSpec
from ..spec.backbones import MINILM
from ..spec.compiler import train_cfg
from .spec import Node, SystemSpec

LABEL_SCAN_ROWS = 20000
TRAIN_ROWS_PER_SPLIT = 200000
OBJECTIVE_TARGET = 0.9
ROLE_TASK = {"router": "classify", "tagger": "frame"}
ROLE_METRIC = {"router": "accuracy", "tagger": "span_f1"}
ROLE_BACKBONE = {"router": MINILM, "tagger": "hf:microsoft/deberta-v3-xsmall"}


def _definition(name: str, role: str) -> str:
    if role == "router":
        return name.replace("_", " ")
    head, kind = tag_parts(name)
    what = kind.replace("_", " ")
    return {"": "not part of any argument", "B": f"first word of the {what}"}.get(head, f"later word of the {what}")


def _labeled(src: HFSource, split: str, rows: int, lmap: list[list[str]]) -> HFSource:
    return src.model_copy(update={"split": split, "role": "labeled", "max_rows": rows, "label_map": lmap})


def node_spec(node: Node, system: SystemSpec) -> TaskSpec:
    from ..data.hf_source import source_labels

    src = node.source
    names = source_labels(src.model_copy(update={"split": node.train_splits[0]}), LABEL_SCAN_ROWS)
    lmap = [[n, n] for n in names] if node.role == "router" else []
    backbone = node.backbone or ROLE_BACKBONE[node.role]
    suites = [
        EvalSuite(
            name=s.name,
            kind="hf",
            source=_labeled(src, s.split, s.max_items * 2, lmap),
            max_items=s.max_items,
            primary=i == 0,
        )
        for i, s in enumerate(system.evals)
    ]
    return TaskSpec(
        name=node.specialist,
        description=node.task,
        summary=node.task,
        modality="text",
        task_type=ROLE_TASK[node.role],
        labels=[Label(name=n, definition=_definition(n, node.role)) for n in names],
        objective=Objective(metric=ROLE_METRIC[node.role], target=OBJECTIVE_TARGET),
        backbone=backbone,
        data=DataPlan(
            synthetic=False, hf_sources=[_labeled(src, s, TRAIN_ROWS_PER_SPLIT, lmap) for s in node.train_splits]
        ),
        train=train_cfg("text", backbone),
        evals=suites,
        baselines=[],
    )


def _exported(p: Project) -> bool:
    manifest = p.root / "export" / "manifest.json"
    return manifest.exists() and json.loads(manifest.read_text())["experiment"] == p.promoted


def build_node(node: Node, system: SystemSpec, log: Callable[[str], None]) -> Project:
    from ..export import export
    from ..pipeline import build

    p = Project(node.specialist)
    if not p.exists():
        p.ensure()
        p.save_spec(node_spec(node, system))
        log(f"[system] created node {node.specialist} ({node.role})")
    if not p.promoted:
        build(node.specialist, log=log)
    if not _exported(p):
        export(node.specialist, log=log)
    return p
