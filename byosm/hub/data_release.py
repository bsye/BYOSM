from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..data.store import BASE_TRAIN, Dataset
from ..evals.test_sets import held_out_set, test_sets
from ..evals.workbench import OURS
from ..model_refs import model_ref
from .facts import CardFacts

SUFFIX = "-data"
UNKNOWN_LICENCE = "licence not stated on its card"
FIELDS = ("id", "input", "label", "source")
LLM_SOURCE = "llm:"
HUB_SOURCE = "hf:"
CARD = """---
license: {license}
language: {languages}
task_categories:
- text-classification
pretty_name: {title} data
size_categories:
- {size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
{extra}---

# {title}: training and test data

The data behind [{model}](https://huggingface.co/{model}), built with [BYOSM](https://github.com/bsye/BYOSM).
Every row has a `source`: `hf:<dataset>` for rows taken from a public dataset, `llm:<model>` for rows an LLM wrote and a
second LLM checked. The `test` split and every extra test set carry `answers`: the label each model gave, scored by the
same code.

| split | rows | sources |
|---|---|---|
{rows}

## Licences

Rows written for this project are {license}. Rows from public datasets keep their own licence:
{sources}

## Labels

{labels}
"""


def data_repo(model_repo: str) -> str:
    return model_repo + SUFFIX


def _answers(f: CardFacts, s: dict) -> dict[str, dict[str, str]]:
    names = {m: (f.exp if m == OURS else model_ref(f.p, m)["name"]) for m in s["results"]}
    return {names[m]: {i: a["label"] for i, a in got.items()} for m, got in s["results"].items() if got}


def _with_answers(f: CardFacts, s: dict) -> list[dict]:
    given = _answers(f, s)
    return [{**r, "answers": {m: got.get(r["id"]) for m, got in given.items()}} for r in s["rows"]]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def splits(f: CardFacts) -> dict[str, list[dict]]:
    trained = Dataset(f.p.data_dir_for(f.spec))
    names = f.meta.get("train_splits") or BASE_TRAIN
    train = [{**{k: r.get(k) for k in FIELDS}, "split_origin": n} for n in names for r in trained.load(n)]
    held = held_out_set(f.p, f.base)
    bench = Dataset(f.p.data_dir_for(f.base))
    out = {
        "train": train,
        "validation": [{k: r.get(k) for k in FIELDS} for r in bench.load("dev")],
        "test": _with_answers(f, held) if held else [{k: r.get(k) for k in FIELDS} for r in bench.load("test")],
    }
    out |= {s["suite"]: _with_answers(f, s) for s in test_sets(f.p, f.base) if not s["board"] and s["rows"]}
    return out


def _source_lines(found: dict[str, list[dict]]) -> str:
    hub = sorted(
        {
            r["source"][len(HUB_SOURCE) :]
            for rows in found.values()
            for r in rows
            if str(r["source"]).startswith(HUB_SOURCE)
        }
    )
    return "\n".join(f"- [`{d}`](https://huggingface.co/datasets/{d}): {_licence(d)}" for d in hub) or "- none"


def _licence(dataset: str) -> str:
    from huggingface_hub import HfApi

    try:
        card = HfApi().dataset_info(dataset).card_data
    except Exception:
        return UNKNOWN_LICENCE
    return (card.get("license") if card else None) or UNKNOWN_LICENCE


def _summary(rows: list[dict]) -> str:
    c = Counter(str(r["source"]) for r in rows)
    return ", ".join(f"{s.removeprefix(LLM_SOURCE)} {n:,}" for s, n in c.most_common())


def stage(f: CardFacts, out: Path) -> Path:
    found = splits(f)
    for name, rows in found.items():
        _write(out / ("data" if name in ("train", "validation", "test") else "test-sets") / f"{name}.jsonl", rows)
    extra = "".join(
        f"- config_name: {n}\n  data_files:\n  - split: test\n    path: test-sets/{n}.jsonl\n"
        for n in found
        if n not in ("train", "validation", "test")
    )
    total = sum(len(r) for r in found.values())
    (out / "README.md").write_text(
        CARD.format(
            license=f.license,
            languages="\n".join(f"- {lang}" for lang in f.base.languages),
            title=f.base.title or f.base.name,
            size="10K<n<100K" if total >= 10_000 else "1K<n<10K",
            extra=extra,
            model=f.repo_id,
            rows="\n".join(f"| {n} | {len(r):,} | {_summary(r)} |" for n, r in found.items()),
            sources=_source_lines(found),
            labels="\n".join(f"- `{lab.name}`: {lab.definition}" for lab in f.base.labels),
        )
    )
    return out


def publish_data(f: CardFacts, out: Path, private: bool, log: Callable[[str], None]) -> str:
    from huggingface_hub import HfApi

    repo = data_repo(f.repo_id)
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="dataset", commit_message=f"BYOSM data for {f.exp}")
    log(f"  published https://huggingface.co/datasets/{repo} ({'private' if private else 'public'})")
    return repo


def releasable(f: CardFacts) -> bool:
    return f.spec.modality == "text"
