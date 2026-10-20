from __future__ import annotations

from statistics import mean, stdev

import yaml

from ..config import Project
from ..spec import TaskSpec
from .cells import STALE, fold_note, folded, stale_ids, stale_note, train_rows

SEEDS = "seeds"
SEED_SEP = "-s"
MIN_SEEDS = 2
IGNORED = {"seed"}


def _group(pid: str) -> str | None:
    head, sep, tail = pid.rpartition(SEED_SEP)
    return head if sep and head and tail.isdigit() else None


def groups(p: Project, suite: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pid, by in p.board().items():
        g = _group(pid)
        if g and by.get(suite, {}).get("kind") == "experiment":
            out.setdefault(g, []).append(pid)
    return {g: sorted(ids) for g, ids in out.items() if len(ids) >= MIN_SEEDS}


def _train(p: Project, exp: str) -> dict:
    return yaml.safe_load((p.exp_dir(exp) / "spec.yaml").read_text()).get("train") or {}


def _varying(settings: dict[str, dict]) -> list[str]:
    keys = sorted({k for t in settings.values() for k in t} - IGNORED)
    return [k for k in keys if len({repr(t.get(k)) for t in settings.values()}) > 1]


def _spread(values: list[float]) -> str:
    return f"{100 * mean(values):.2f} ± {100 * stdev(values):.2f}"


def _row(p: Project, g: str, ids: list[str], by: dict, cols: list[str], train: dict, stale: set[str]) -> str:
    name = f"**{g}**" if p.promoted in ids else g
    name += STALE if stale & set(ids) else ""
    cells = [name, str(len(ids)), " / ".join(sorted({train_rows(p, i) for i in ids}))]
    cells += ["-" if train.get(k) is None else str(train[k]).lower() for k in cols]
    cells.append(_spread([by[i]["objective"] for i in ids]))
    cells.append(_spread([by[i]["calib_objective"] for i in ids]))
    return "| " + " | ".join(cells) + " |"


def _note(prim: str, top: str, any_folded: bool, stale: set[str]) -> str:
    return (
        f"Experiments named `<recipe>{SEED_SEP}<seed>` on suite `{prim}`, mean ± sample sd over seeds, sorted by calib "
        f"(the selection split); top on calib: `{top}`. Columns after train rows are the `train` settings that differ "
        "between recipes; - = not set (the code default when it ran)." + fold_note(any_folded) + stale_note(stale)
    )


def seeds_block(name: str) -> str:
    p = Project(name)
    spec = TaskSpec.load(p.spec_path)
    prim = spec.primary_suite()
    found = groups(p, prim)
    if not found:
        return ""
    by = {pid: r[prim] for pid, r in p.board().items() if prim in r}
    settings = {g: _train(p, ids[0]) for g, ids in found.items()}
    cols = _varying(settings)
    order = sorted(found, key=lambda g: -mean(by[i]["calib_objective"] for i in found[g]))
    stale = stale_ids(p, prim)
    metric = spec.objective.metric
    names = ["recipe", "seeds", "train rows", *cols, f"test {metric} %", f"calib {metric} %"]
    head = "| " + " | ".join(names) + " |"
    rule = "|---" * len(names) + "|"
    body = [_row(p, g, found[g], by, cols, settings[g], stale) for g in order]
    any_folded = any(folded(p, i) for ids in found.values() for i in ids)
    return "\n".join([head, rule, *body, "", _note(prim, order[0], any_folded, stale)])
