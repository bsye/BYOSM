from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..ablation import summaries as ablation_summaries
from ..archive import archived, set_archived
from ..auto.standing import best_baseline
from ..data.store import STORED_SPLITS, Dataset
from ..data.sufficiency import data_check
from ..dataset_search import saved_suggestions
from ..evals.overfit import OVERFIT
from ..evals.scoring import busy
from ..health import health
from ..hub.facts import suite_dataset
from ..improve.bandit import search_summary
from ..improve.proposals import PROPOSALS
from ..lab.contract import SPECIALIST_MODALITIES
from ..metrics import metric_label, unreached
from ..model_refs import model_refs
from ..next_steps import suggestions, suggestions_or_none
from ..project import Project, list_projects
from ..replicas import seed_spread
from ..rungs import auxiliary
from ..spec import Objective, TaskSpec

MAX_PAGE = 500
RESULT_FIELDS = ("objective", "calib_objective", "latency_ms", "params", "n_test", "n_calib")
SUITE_FIELDS = ("name", "kind", "primary", "written_by", "judged_by")
META_FIELDS = ("params", "n_train", "train_seconds", "gpu_hours", "best_epoch")


def project(ws: Path, name: str) -> Project:
    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    return p


def _result(o: Objective, r: dict) -> dict[str, Any]:
    m = r.get("metrics", {})
    per_label = m.get("per_label", {})
    caught = {k: v.get("recall") for k, v in per_label.items()} if len(per_label) == 2 else None
    shaped = {k: r.get(k) for k in RESULT_FIELDS} | {
        "ci": m.get("ci95"),
        "unreached": unreached(o, m),
        "caught": caught,
    }
    return shaped | {"latency_load": r.get("latency_load"), "timed_busy": busy(r)}


def _benchmark(spec: TaskSpec, suite: str) -> str:
    kind = next((e.kind for e in spec.evals if e.name == suite), None)
    labeled = [h for h in spec.data.hf_sources if h.role == "labeled"]
    if kind == "builtin" and labeled:
        return f"{labeled[0].dataset.rpartition('/')[2]}, held out"
    return suite_dataset(spec, suite)[0].rpartition("/")[2]


def summary(ws: Path, name: str) -> dict[str, Any]:
    from ..export.bundle import is_exported

    p = project(ws, name)
    s = p.spec()
    prim = s.primary_suite()
    board = p.board()
    best = board.get(p.promoted or "", {}).get(prim, {})
    rival = best_baseline(board, prim)
    return {
        "archived": name in archived(ws),
        "name": name,
        "title": s.title,
        "created": s.created.isoformat() if s.created else None,
        "description": s.description,
        "modality": s.modality,
        "task_type": s.task_type,
        "labels": len(s.labels),
        "metric": metric_label(s.objective),
        "primary_suite": prim,
        "benchmark": _benchmark(s, prim),
        "promoted": p.promoted,
        "score": best.get("objective"),
        "rival": {"name": rival[0], "score": board[rival[0]][prim].get("objective")} if rival else None,
        "experiments": p.experiments(),
        "exported": is_exported(p.root / "export"),
    }


def specialists(ws: Path) -> list[dict[str, Any]]:
    from ..series.board import reports
    from ..series.specialist import summary as forecaster
    from ..tabular.specialist import summary as risk
    from ..tabular.task import forecasts
    from .teams import members, team_names, team_summary

    part_of, shelved = members(ws), archived(ws)
    every = (
        [{**summary(ws, n), "kind": "model", "part_of": part_of.get(n)} for n in list_projects(ws)]
        + [team_summary(ws, t) for t in team_names(ws)]
        + [forecaster(r) for r in reports(ws) if r["at"]]
        + [risk(r) for r in forecasts(ws)]
    )
    return [{**s, "archived": s["name"] in shelved} for s in every]


def _run(p: Project, exp: str) -> dict[str, Any]:
    e = p.experiment(exp) or {}
    m = e.get("meta") or {}
    return {
        "name": exp,
        "parent": (e.get("proposal") or {}).get("parent"),
        "continued": ((e.get("spec") or {}).get("train") or {}).get("init_from"),
        "created": e["created"].timestamp() if e.get("created") else None,
        "auxiliary": auxiliary(exp),
        **{k: m.get(k) for k in META_FIELDS},
        "history": [
            {"epoch": h["epoch"] + 1, "loss": h["loss"], "dev": h["dev_objective"]} for h in m.get("history", [])
        ],
    }


def detail(ws: Path, name: str) -> dict[str, Any]:
    p = project(ws, name)
    spec = p.spec()
    board = {pid: {suite: _result(spec.objective, r) for suite, r in by.items()} for pid, by in p.board().items()}
    data = Dataset(p.data_dir_for(spec))
    return {
        **summary(ws, name),
        "spec": spec.model_dump(mode="json"),
        "board": board,
        "models": model_refs(p, p.board()),
        "suites": [{k: getattr(e, k) for k in SUITE_FIELDS} for e in spec.evals],
        "baselines": spec.baselines,
        "runs": [_run(p, e) for e in p.experiments()],
        "proposals": p.report(PROPOSALS),
        "search": search_summary(p),
        "data": data.summary() if data.built() else None,
        "overfit": p.report(OVERFIT),
        "next": suggestions(name, ws),
        "data_check": data_check(p, spec),
        "health": health(p, spec),
        "data_suggestions": saved_suggestions(p),
        "ablations": ablation_summaries(p),
        "seeds": seed_spread(p),
        "lab": spec.modality in SPECIALIST_MODALITIES,
    }


def health_of(ws: Path, name: str) -> dict[str, Any] | None:
    from ..series.health import of_domain

    return found if (found := of_domain(ws, name)) is not None else health(p := project(ws, name), p.spec())


def steps_of(ws: Path, name: str) -> list[dict[str, Any]]:
    from ..series.steps import of_domain

    return found if (found := of_domain(ws, name)) is not None else suggestions(project(ws, name).name, ws)


def next_steps(ws: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: suggestions_or_none(name, ws) for name in list_projects(ws)}


def save_spec(ws: Path, name: str, body: dict) -> dict[str, Any]:
    p = project(ws, name)
    spec = TaskSpec(**{**body, "name": name})
    p.save_spec(spec)
    return spec.model_dump(mode="json")


def delete(ws: Path, name: str) -> None:
    p = project(ws, name)
    set_archived(ws, name, False)
    p.delete()
    shutil.rmtree(p.root, ignore_errors=True)


def rows(ws: Path, name: str, split: str, offset: int, limit: int) -> dict[str, Any]:
    if split not in STORED_SPLITS:
        raise KeyError(f"split must be one of {', '.join(STORED_SPLITS)}")
    data = Dataset((p := project(ws, name)).data_dir_for(p.spec())).load(split)
    return {"total": len(data), "rows": data[offset : offset + min(limit, MAX_PAGE)]}


def systems(ws: Path) -> list[dict[str, Any]]:
    from .teams import team_names

    return [system(ws, n) for n in team_names(ws)]


def system(ws: Path, name: str) -> dict[str, Any]:
    from ..system import SystemProject
    from ..system.board import BOARD

    sp = SystemProject(name, ws)
    if not sp.exists():
        raise KeyError(f"no system {name!r}")
    return {
        "name": name,
        "spec": sp.spec().model_dump(mode="json"),
        "board": sp.report(BOARD),
        "exported": (sp.export_dir / "system.json").exists(),
    }
