from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .data.store import Dataset
from .project import Project
from .spec import TaskSpec
from .text import match_label

MAX_RIVALS = 2
MAX_PARAMS = 1_000_000_000
MIN_COVERAGE = 0.6
MIN_AGREEMENT = 0.8
SEARCH_LIMIT = 8
TEXT_CLASSIFIER = "text-classification"
REPORT = "finetuned-rivals"
GENERIC_LABEL = "LABEL_"


def benchmarks(spec: TaskSpec) -> list[str]:
    prim = next((e for e in spec.evals if e.name == spec.primary_suite()), None)
    found = [prim.source.dataset] if prim and prim.source else []
    found += [s.dataset for s in spec.data.hf_sources if s.role == "labeled"]
    names = [*found, *(d.rpartition("/")[2] for d in found)]
    return list(dict.fromkeys(names))


def _size(repo: str) -> int:
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, files_metadata=False)
    except Exception:
        return 0
    return int(getattr(getattr(info, "safetensors", None), "total", 0) or 0)


def candidates(spec: TaskSpec, log: Callable[[str], None] = print) -> list[str]:
    from huggingface_hub import HfApi

    from .config import load_config
    from .data.hf_source.research import research_models

    picked = [m["id"] for m in research_models(spec, benchmarks(spec), load_config(), log)]
    seen: dict[str, int] = {}
    for ds in benchmarks(spec):
        try:
            found = HfApi().list_models(
                trained_dataset=ds, pipeline_tag=TEXT_CLASSIFIER, sort="downloads", limit=SEARCH_LIMIT
            )
            seen |= {m.id: getattr(m, "downloads", 0) or 0 for m in found if m.id not in seen}
        except Exception:
            continue
    ranked = list(dict.fromkeys([*picked, *sorted(seen, key=lambda r: -seen[r])]))
    return [r for r in ranked if _size(r) <= MAX_PARAMS]


def _rival(p: Project, spec: TaskSpec, repo: str) -> tuple[str, str] | None:
    from .discover import _model_labels
    from .predictors.label_map import suggest_label_map

    labels = _model_labels(repo) or []
    if not labels:
        return None
    if not all(x.startswith(GENERIC_LABEL) for x in labels):
        covered = {match_label(x, spec.label_names) for x in labels} - {None}
        share = len(covered) / max(1, len(spec.labels))
        return (
            (f"hf:{repo}", f"fine-tuned on the benchmark, names {share:.0%} of our labels")
            if share >= MIN_COVERAGE
            else None
        )
    hint = suggest_label_map(spec, Dataset(p.data_dir_for(spec)).load("dev"), repo)
    if hint["agreement"] < MIN_AGREEMENT:
        return None
    pairs = ",".join(f"{k}={v}" for k, v in hint["map"].items())
    return (
        f"hf:{repo}#{pairs}",
        f"fine-tuned on the benchmark, its outputs map onto ours on {hint['agreement']:.0%} of our rows",
    )


def add_finetuned(name: str, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    p = Project(name)
    spec = p.spec()
    added: list[dict[str, Any]] = []
    for repo in candidates(spec, log):
        if len(added) >= MAX_RIVALS:
            break
        try:
            found = _rival(p, spec, repo)
        except Exception as e:
            log(f"  skip {repo}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if found is None:
            log(f"  skip {repo}: its labels do not map onto ours")
            continue
        added.append({"rival": found[0], "why": found[1]})
        log(f"  {found[0]}: {found[1]}")
    spec.baselines = [*spec.baselines, *(a["rival"] for a in added if a["rival"] not in spec.baselines)]
    p.save_spec(spec)
    p.save_report(REPORT, {"benchmarks": benchmarks(spec), "added": added})
    return added
