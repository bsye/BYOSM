from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace

from ..data.judge import DROP, JUDGE_BATCH, JUDGE_ROLE, judge_labels
from ..data.prompts import task_context
from ..llm import parallel_calls, role_model
from ..llm.memo import memo_key, memoised
from ..spec import TaskSpec
from .layouts import marked
from .pairs import Pair
from .rewrite import flat

RELABEL_VERSION = "1"
NEAR_MISS_SOURCE = "near-miss:"

Log = Callable[[str], None]


def relabel(spec: TaskSpec, pairs: list[Pair], cfg: dict | None, log: Log) -> list[Pair]:
    if not pairs:
        return []
    model = role_model(JUDGE_ROLE, cfg)
    texts = [flat(marked(p.request, p.option, p.offered)) for p in pairs]
    keys = [memo_key(RELABEL_VERSION, model, task_context(spec), t) for t in texts]
    text_of = dict(zip(keys, texts, strict=True))

    def compute(part: list[str]) -> dict[str, str]:
        labels = judge_labels(spec, [text_of[k] for k in part], cfg, 1, log)
        return {k: lab for k, lab in zip(part, labels, strict=True) if lab}

    found = memoised(keys, compute, JUDGE_BATCH, parallel_calls(cfg), log)
    source = f"{NEAR_MISS_SOURCE}{model}"
    kept = [replace(p, label=lab, source=source) for p, lab in zip(pairs, found, strict=True) if lab and lab != DROP]
    tally = Counter((p.kind, p.label) for p in kept)
    log(f"  near misses: {len(kept)} of {len(pairs)} kept after relabelling by {model}: {dict(tally)}")
    return kept
