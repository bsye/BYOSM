from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CPU, GPU = "cpu", "gpu"
MODEL, FORECAST = "model", "forecast"
PRETRAINING = "pretraining-v2"
FORECAST_ROUNDS = 2


@dataclass(frozen=True)
class Measurement:
    id: str
    title: str
    lane: str


MEASUREMENTS = {
    m.id: m
    for m in (
        Measurement("overfit", "Overfitting", CPU),
        Measurement("public-suites", "Public benchmarks", CPU),
        Measurement("seeds", "Seed noise", GPU),
        Measurement("curve", "Learning curve", GPU),
        Measurement(PRETRAINING, "Pretraining v2", GPU),
    )
}
ORDER = list(MEASUREMENTS)


def columns() -> list[dict[str, str]]:
    return [asdict(m) for m in MEASUREMENTS.values()]


def _models(ws: Path, names: list[str] | None) -> list[str]:
    from ..archive import archived
    from ..project import list_projects

    shelved = archived(ws)
    return [n for n in list_projects(ws) if n not in shelved and (names is None or n in names)]


def _forecasters(ws: Path, names: list[str] | None) -> list[dict[str, Any]]:
    from ..archive import archived
    from ..series.board import reports

    shelved = archived(ws)
    found = [r for r in reports(ws) if r["at"] and r.get("promoted") and r["domain"] not in shelved]
    return [r for r in found if names is None or r["domain"] in names]


def known(ws: Path) -> set[str]:
    return {*_models(ws, None), *(r["domain"] for r in _forecasters(ws, None))}


def _forecast_step(report: dict[str, Any]) -> dict[str, Any] | None:
    from ..series.escalation import tried
    from ..series.patch import pretrained
    from ..series.recipe import V2

    if tried(report, V2):
        return None
    found = {"id": PRETRAINING, "model": report["promoted"]}
    if pretrained(V2) is None:
        why = "No forecaster was trained on the v2 recipe yet. Pretrains its base once, for every domain."
        return {**found, "why": why, "args": ["series", "pretrain", "--recipe", V2.name]}
    domain = report["domain"]
    why = f"No version of {domain} was trained on the v2 base. Adds versions until one beats the best rival; v2 comes once more training stops helping."
    return {**found, "why": why, "args": ["series", "run", domain, "--until-rival", "--rounds", str(FORECAST_ROUNDS)]}


def steps(ws: Path, specialist: str, kind: str) -> list[dict[str, Any]]:
    from ..next_steps import suggestions_or_none
    from ..project import Project

    if kind == FORECAST:
        return [s for r in _forecasters(ws, [specialist]) if (s := _forecast_step(r))]
    model = Project(specialist, ws).promoted
    return [{**s, "model": model} for s in suggestions_or_none(specialist, ws) if s["id"] in MEASUREMENTS]


def cell(specialist: str, kind: str, step: dict[str, Any]) -> dict[str, Any]:
    m = MEASUREMENTS[step["id"]]
    return {
        "specialist": specialist,
        "kind": kind,
        "measurement": m.id,
        "title": m.title,
        "lane": m.lane,
        "model": step["model"],
        "why": step["why"],
        "args": step["args"],
    }


def ordered(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = {c["specialist"]: i for i, c in reversed(list(enumerate(cells)))}

    def rank(c: dict[str, Any]) -> tuple[int, int, int]:
        who, what = first[c["specialist"]], ORDER.index(c["measurement"])
        return (0, what, who) if c["lane"] == CPU else (1, who, what)

    return sorted(cells, key=rank)


def plan(ws: Path, names: list[str] | None = None) -> list[dict[str, Any]]:
    if names and (unknown := sorted(set(names) - known(ws))):
        raise KeyError(f"no specialist or forecaster named {', '.join(unknown)}")
    models = [cell(n, MODEL, s) for n in _models(ws, names) for s in steps(ws, n, MODEL)]
    forecasts = [cell(r["domain"], FORECAST, s) for r in _forecasters(ws, names) if (s := _forecast_step(r))]
    return ordered(models + forecasts)
