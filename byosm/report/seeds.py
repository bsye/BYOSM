from __future__ import annotations

from statistics import mean, stdev

from ..project import Project
from .cells import STALE, fold_note, folded, stale_ids, stale_note, train_rows

SEEDS = "seeds"
SEED_SEP = "-s"
MIN_SEEDS = 2
IGNORED = {"seed"}


def _group(pid: str) -> str | None:
    head, sep, tail = pid.rpartition(SEED_SEP)
    return head if sep and head and tail.isdigit() else None


def scored(p: Project, suite: str) -> dict[str, dict]:
    return {pid: by[suite] for pid, by in p.board().items() if by.get(suite, {}).get("kind") == "experiment"}


def _by_recipe_and_fingerprint(by: dict[str, dict]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for pid in sorted(by):
        g = _group(pid)
        if g:
            out.setdefault(g, {}).setdefault(str(by[pid].get("fingerprint")), []).append(pid)
    return out


def recipes(by: dict[str, dict]) -> dict[str, list[str]]:
    out = {}
    for g, subsets in _by_recipe_and_fingerprint(by).items():
        kept = {fp: ids for fp, ids in subsets.items() if len(ids) >= MIN_SEEDS}
        out |= {g if len(kept) == 1 else f"{g} @{fp}": ids for fp, ids in kept.items()}
    return out


def _train(p: Project, exp: str) -> dict:
    return p.exp_spec(exp).model_dump(mode="json").get("train") or {}


def _varying(settings: dict[str, dict]) -> list[str]:
    keys = sorted({k for t in settings.values() for k in t} - IGNORED)
    return [k for k in keys if len({repr(t.get(k)) for t in settings.values()}) > 1]


def _spread(values: list[float]) -> str:
    head = f"{100 * mean(values):.2f}"
    return f"{head} ± {100 * stdev(values):.2f}" if len(values) >= MIN_SEEDS else head


def _row(p: Project, g: str, ids: list[str], by: dict, cols: list[str], train: dict, stale: set[str]) -> str:
    name = f"**{g}**" if p.promoted in ids else g
    name += STALE if stale & set(ids) else ""
    cells = [name, str(len(ids)), " / ".join(sorted({train_rows(p, i) for i in ids}))]
    cells += ["-" if train.get(k) is None else str(train[k]).lower() for k in cols]
    cells.append(_spread([by[i]["objective"] for i in ids]))
    cells.append(_spread([by[i]["calib_objective"] for i in ids]))
    return "| " + " | ".join(cells) + " |"


def _top(top: str | None) -> str:
    return f"top on calib: `{top}`" if top else "no recipe has seeds on the promoted model's suite version"


def _note(prim: str, top: str | None, any_folded: bool, stale: set[str]) -> str:
    return (
        f"Experiments named `<recipe>{SEED_SEP}<seed>` on suite `{prim}`, mean ± sample sd over seeds, sorted by calib "
        "(the selection split); only seeds scored on the same suite version are averaged (a recipe scored on several "
        "gets one row per version, `@<fingerprint>`; a seed alone on its version stays in the main table; "
        f"{STALE} rows are listed last and never ranked against the promoted version); {_top(top)}. Columns after train rows are the `train` settings that differ "
        "between recipes; - = not set (the code default when it ran)." + fold_note(any_folded) + stale_note(stale)
    )


def seeds_block(name: str) -> str:
    p = Project(name)
    spec = p.spec()
    prim = spec.primary_suite()
    by = scored(p, prim)
    found = recipes(by)
    if not found:
        return ""
    settings = {g: _train(p, ids[0]) for g, ids in found.items()}
    cols = _varying(settings)
    stale = stale_ids(p, prim)
    order = sorted(
        found, key=lambda g: (bool(stale & set(found[g])), -mean(by[i]["calib_objective"] for i in found[g]))
    )
    top = next((g for g in order if not stale & set(found[g])), None)
    metric = spec.objective.metric
    names = ["recipe", "seeds", "train rows", *cols, f"test {metric} %", f"calib {metric} %"]
    head = "| " + " | ".join(names) + " |"
    rule = "|---" * len(names) + "|"
    body = [_row(p, g, found[g], by, cols, settings[g], stale) for g in order]
    any_folded = any(folded(p, i) for ids in found.values() for i in ids)
    return "\n".join([head, rule, *body, "", _note(prim, top, any_folded, stale)])
