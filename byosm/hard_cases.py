from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .data.store import NEAR_DUPLICATE, Dataset, ShingleIndex, record, refine_split
from .project import Project
from .spec import PairPlan, TaskSpec

POOL_REQUESTS = 4000
HARD_ROWS = 3000
HARD_STEP = "hard-cases"
HARD_KIND = "hard-case"
SCORE_CHUNK = 256
Log = Callable[[str], None]


def _held_rows(p: Project, base: TaskSpec) -> list[dict]:
    from .evals.suites import stored

    out: list[dict] = []
    for suite in base.evals:
        if found := stored(p, base, suite):
            out += found[0] + found[1]
    return out


def pool_rows(p: Project, base: TaskSpec, pool: str, log: Log) -> list[dict]:
    from .hub.facts import HUB_SOURCE
    from .options.pairs import pairs_of, rendered, split_layouts
    from .pools import POOLS

    _, cases, dataset = POOLS[pool]
    layouts, mask = split_layouts(base.data.pairs or PairPlan(), "train")
    held = ShingleIndex(_held_rows(p, base))
    rows, dropped = [], 0
    for c in cases():
        if held.contained(c.situation.text) >= NEAR_DUPLICATE:
            dropped += 1
            continue
        for pair in pairs_of(c.group, c.situation, HUB_SOURCE + dataset, HARD_KIND):
            text, layout = rendered(pair, layouts, mask)
            meta = {"kind": HARD_KIND, "group": c.group, "category": c.category, "layout": layout}
            rows.append(record(text, pair.label, pair.source, **meta))
    log(f"[hard] {pool}: {len(rows)} pairs; {dropped} requests left out as near copies of a calib or test row")
    return rows


def _sample(rows: list[dict]) -> list[dict]:
    groups = sorted({r["meta"]["group"] for r in rows})
    random.Random(0).shuffle(groups)
    kept = set(groups[:POOL_REQUESTS])
    return [r for r in rows if r["meta"]["group"] in kept]


def _losses(p: Project, base: TaskSpec, parent: str, rows: list[dict], log: Log) -> np.ndarray:
    from .evals.board import experiment_predictor
    from .events import emit

    pred = experiment_predictor(p, base, parent)
    probs = []
    for start in range(0, len(rows), SCORE_CHUNK):
        probs.append(pred.probs_batch([r["input"] for r in rows[start : start + SCORE_CHUNK]]))
        emit("scoring", predictor=parent, suite=HARD_STEP, split="pool", done=start + len(probs[-1]), total=len(rows))
    gold = np.array([base.label_index[r["label"]] for r in rows])
    found = np.concatenate(probs)[np.arange(len(rows)), gold]
    log(f"[hard] scored {len(rows)} pool pairs with {parent}")
    return -np.log(np.clip(found, 1e-9, 1.0))


def mine(p: Project, base: TaskSpec, parent: str, rows: list[dict], log: Log) -> list[dict]:
    sample = _sample(rows)
    loss = _losses(p, base, parent, sample, log)
    hard = [sample[i] for i in np.argsort(-loss)[:HARD_ROWS]]
    kinds = Counter((r["meta"]["category"], r["label"]) for r in hard)
    log(f"[hard] kept the {len(hard)} pairs {parent} gets most wrong: {dict(kinds)}")
    return hard


def spent(p: Project) -> str | None:
    from .improve.bandit import VERDICT

    verdicts = p.reports(VERDICT)
    for exp in p.attempts():
        proposal = p.proposal(exp) or {}
        ours = any(a.get("id") == HARD_STEP for a in proposal.get("applied", []))
        if ours and proposal.get("parent") == p.promoted and (verdicts.get(VERDICT + exp) or {}).get("won") is False:
            return exp
    return None


def hard_round(name: str, log: Log = print) -> dict[str, Any]:
    from .pipeline import build, next_exp
    from .pools import pools_for
    from .train.warm import continued

    p = Project(name)
    base, parent = p.spec(), p.promoted
    pools = pools_for(base)
    if not parent or not pools:
        raise RuntimeError(f"{name} needs a promoted model and a pool of unscored rows that fits it")
    if lost := spent(p):
        raise RuntimeError(f"{lost} already added hard cases to {parent} and did not win; the next lever is new data")
    hard = mine(p, base, parent, [r for pool in pools for r in pool_rows(p, base, pool, log)], log)
    exp, spec = next_exp(p), continued(p.exp_spec(parent), parent)
    data = Dataset(p.data_dir_for(spec))
    earlier = data.load(refine_split(parent))
    data.save(refine_split(exp), earlier + hard)
    why = f"{len(hard)} pool rows {parent} gets most wrong ({', '.join(pools)}) on top of {len(earlier)} earlier ones"
    proposal = {"parent": parent, "applied": [{"id": HARD_STEP, "why": why, "ops": []}]}
    p.save_experiment(exp, spec=spec, proposal=proposal)
    log(f"created {name}/{exp}: {why}, warm from {parent}")
    return build(name, exp, log)
