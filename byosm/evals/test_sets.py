from __future__ import annotations

from typing import Any

from ..data.store import Dataset
from ..project import Project
from ..spec import TaskSpec
from .testset import written_logs
from .workbench import OURS, results

ROW_FIELDS = ("id", "input", "label", "source")
HELD_OUT = "held out"


def _answers(p: Project, spec: TaskSpec, pid: str) -> dict[str, dict]:
    r = p.result(pid, spec.primary_suite()) or {}
    units = (r.get("units") or {}).get("test") or {}
    rows = Dataset(p.data_dir_for(spec)).load("test")
    if len(units.get("pred", [])) != len(rows):
        return {}
    return {
        row["id"]: {"label": spec.label_names[k], "confidence": None}
        for row, k in zip(rows, units["pred"], strict=True)
    }


def held_out_set(p: Project, spec: TaskSpec) -> dict[str, Any] | None:
    if spec.task_type != "classify" or next(e.kind for e in spec.evals if e.name == spec.primary_suite()) != "builtin":
        return None
    rows = Dataset(p.data_dir_for(spec)).load("test")
    models = {OURS: p.promoted, **{b: b for b in spec.baselines}}
    found = {key: _answers(p, spec, pid) for key, pid in models.items() if pid}
    shown = [{k: r.get(k) for k in ROW_FIELDS} for r in rows]
    return {
        "suite": HELD_OUT,
        "frozen": True,
        "board": True,
        "rows": shown,
        "results": {k: v for k, v in found.items() if v},
    }


def test_sets(p: Project, spec: TaskSpec) -> list[dict[str, Any]]:
    frozen = {e.name for e in spec.evals}
    written = [
        {
            "suite": s,
            "frozen": s in frozen,
            "board": False,
            "rows": [{k: r.get(k) for k in ROW_FIELDS} for r in rows],
            "results": results(p, s),
        }
        for s, rows in written_logs(p).items()
    ]
    held = held_out_set(p, spec)
    return [*written, *([held] if held else [])]


MAX_PAGE = 500


def _score(s: dict[str, Any], model: str) -> dict[str, int]:
    labels = {r["id"]: r["label"] for r in s["rows"]}
    got = s["results"].get(model, {})
    scored = [i for i in got if i in labels]
    return {"right": sum(got[i]["label"] == labels[i] for i in scored), "scored": len(scored)}


def same_family(s: dict[str, Any]) -> dict[str, str]:
    from ..data.budget import LLM_SOURCE
    from ..model_refs import family, llm_model

    sources = {str(r.get("source", "")) for r in s["rows"]}
    writers = sorted({w.rpartition(":")[2] for w in sources if w.startswith(LLM_SOURCE)})
    models = {m: llm_model(m) for m in s["results"]}
    return {m: w for m, model in models.items() if model for w in writers if family(w) == family(model)}


def summaries(p: Project, spec: TaskSpec) -> list[dict[str, Any]]:
    return [
        {
            "suite": s["suite"],
            "frozen": s["frozen"],
            "board": s["board"],
            "total": len(s["rows"]),
            "scores": {m: _score(s, m) for m in s["results"]},
            "same_family": same_family(s),
        }
        for s in test_sets(p, spec)
    ]


def page(p: Project, spec: TaskSpec, suite: str, offset: int, limit: int) -> dict[str, Any]:
    s = next((t for t in test_sets(p, spec) if t["suite"] == suite), None)
    if s is None:
        raise KeyError(f"no test set {suite!r}")
    rows = list(reversed(s["rows"]))[offset : offset + min(limit, MAX_PAGE)]
    ids = {r["id"] for r in rows}
    results = {m: {i: a for i, a in got.items() if i in ids} for m, got in s["results"].items()}
    return {"total": len(s["rows"]), "rows": rows, "results": results}
