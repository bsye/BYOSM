from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .data.store import Dataset
from .project import Project
from .spec import TaskSpec
from .text import match_label

MAX_RIVALS = 3
MAX_PARAMS = 1_000_000_000
MIN_COVERAGE = 0.6
MIN_AGREEMENT = 0.8
SEARCH_LIMIT = 8
PIPELINES = {
    ("text", "classify"): "text-classification",
    ("text", "frame"): "token-classification",
    ("audio", "classify"): "audio-classification",
    ("image", "classify"): "image-classification",
    ("document", "classify"): "image-classification",
}
SCORE_METRICS = ("accuracy", "f1")
OWN_TAG = "byosm"
REPORT = "finetuned-rivals"
GENERIC_LABEL = "LABEL_"


def benchmarks(spec: TaskSpec) -> list[str]:
    prim = next((e for e in spec.evals if e.name == spec.primary_suite()), None)
    found = [prim.source.dataset] if prim and prim.source else []
    found += [s.dataset for s in spec.data.hf_sources if s.role == "labeled"]
    names = [*found, *(d.rpartition("/")[2] for d in found)]
    return list(dict.fromkeys(names))


def _strength(repo: str, datasets: list[str]) -> tuple[float, int] | None:
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, expand=["cardData", "safetensors", "tags"])
    except Exception:
        return None
    if OWN_TAG in (info.tags or []):
        return None
    reported = [
        float(r.metric_value)
        for r in getattr(info.card_data, "eval_results", None) or []
        if any(d in f"{r.dataset_type} {r.dataset_name}" for d in datasets) and r.metric_type in SCORE_METRICS
    ]
    return max(reported, default=-1.0), int(getattr(getattr(info, "safetensors", None), "total", 0) or 0)


def candidates(spec: TaskSpec, log: Callable[[str], None] = print) -> list[str]:
    from huggingface_hub import HfApi

    from .config import load_config
    from .data.hf_source.research import research_models

    picked = [m["id"] for m in research_models(spec, benchmarks(spec), load_config(), log)]
    tag, sets = PIPELINES.get((spec.modality, spec.task_type)), benchmarks(spec)
    seen: set[str] = set()
    for ds in sets:
        try:
            seen |= {
                m.id
                for m in HfApi().list_models(trained_dataset=ds, pipeline_tag=tag, sort="downloads", limit=SEARCH_LIMIT)
            }
        except Exception:
            continue
    known = {b.removeprefix("hf:").partition("#")[0] for b in spec.baselines}
    strength = {r: s for r in dict.fromkeys([*picked, *seen]) if r not in known and (s := _strength(r, sets))}
    fitting = [r for r, (_, size) in strength.items() if size <= MAX_PARAMS]
    return sorted(fitting, key=lambda r: (-strength[r][0], -strength[r][1]))


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
    p.add_baselines([a["rival"] for a in added])
    p.save_report(REPORT, {"benchmarks": benchmarks(spec), "added": added})
    return added
