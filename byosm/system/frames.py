from __future__ import annotations

import random
from collections import Counter

import numpy as np

from ..data.store import read, write
from ..evals.suites import CALIB_EVERY_NTH_ITEM
from ..metrics import f1_of_counts
from ..resample import bootstrap_stats
from ..rt import spans
from ..text import markup_tags, normalize
from .spec import SystemProject, SystemSpec, SystemSuite

SPLITS = ("calib", "test")
BUILTIN_SPLIT = "builtin"


def _gold(row: dict, spec: SystemSpec, names: list[str] | None) -> dict:
    router, tagger = spec.role("router"), spec.role("tagger")
    action = names[row[router[1].source.label_column]] if router and names else None
    args = []
    if tagger:
        if tagger[1].source.format != "markup":
            raise ValueError("system suites read arguments from inline markup; this tagger's source has none")
        words, tags = markup_tags(row[tagger[1].source.label_column])
        args = [[k, " ".join(words[a:b])] for k, a, b in spans(tags)]
    return {"action": action, "args": args}


def _held_out(spec: SystemSpec, suite: SystemSuite) -> list[dict]:
    from ..data.store import Dataset
    from ..project import Project

    router = spec.role("router")
    if router is None or spec.role("tagger"):
        raise ValueError("a builtin system suite needs a router and no tagger: its rows carry no argument spans")
    p = Project(router[1].specialist)
    rows = Dataset(p.data_dir_for(p.spec())).load("test")[: suite.max_items]
    return [{"input": r["input"], "action": r["label"], "args": []} for r in rows]


def _fetch(spec: SystemSpec, suite: SystemSuite) -> list[dict]:
    import itertools

    if suite.split == BUILTIN_SPLIT:
        return _held_out(spec, suite)

    from ..data.hf_source.hub import open_split
    from ..data.hf_source.rows import class_names

    first = next(iter(spec.nodes.values())).source
    router = spec.role("router")
    ds = open_split(first.dataset, first.config, suite.split)
    names = class_names(router[1].source, ds) if router else None
    rows = [{"input": r[first.input_column], **_gold(r, spec, names)} for r in itertools.islice(ds, suite.max_items)]
    random.Random(0).shuffle(rows)
    return rows


def materialize(sp: SystemProject, suite: SystemSuite) -> dict[str, list[dict]]:
    d = sp.root / "evals" / suite.name
    if not (d / "test.jsonl").exists():
        rows = _fetch(sp.spec(), suite)
        d.mkdir(parents=True, exist_ok=True)
        write(d / "calib.jsonl", rows[::CALIB_EVERY_NTH_ITEM])
        write(d / "test.jsonl", [r for i, r in enumerate(rows) if i % CALIB_EVERY_NTH_ITEM])
    return {s: read(d / f"{s}.jsonl") for s in SPLITS}


def _args(frame: dict) -> Counter:
    return Counter((t, normalize(x)) for t, x in frame["args"])


def units(pred: list[dict], gold: list[dict]) -> dict[str, list]:
    out: dict[str, list] = {"exact": [], "action": [], "args": []}
    for p, g in zip(pred, gold, strict=True):
        pa, ga = _args(p), _args(g)
        tp = sum((pa & ga).values())
        same_action = p["action"] == g["action"]
        out["exact"].append(int(same_action and pa == ga))
        out["action"].append(int(same_action))
        out["args"].append([tp, sum(pa.values()) - tp, sum(ga.values()) - tp])
    return out


def summary(u: dict[str, list]) -> dict[str, float | None]:
    counts = np.asarray(u["args"]).sum(0)
    return {
        "exact_frame": float(np.mean(u["exact"])),
        "action_accuracy": float(np.mean(u["action"])),
        "arg_f1": float(f1_of_counts(counts)) if counts.any() else None,
        "n": len(u["exact"]),
    }


def _hits(u: dict[str, list]) -> np.ndarray:
    return np.stack([np.asarray(u["exact"]), np.ones(len(u["exact"]), int)], 1)


def exact_bootstrap(a: dict[str, list], b: dict[str, list] | None = None) -> dict[str, float]:
    return bootstrap_stats(_hits(a), _hits(b) if b else None, lambda s: float(s[0] / s[1]))
