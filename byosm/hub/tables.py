from __future__ import annotations

from .facts import TASK_NAME, CardFacts, fmt, suite_dataset

DETAIL_HEAD = "| suite | dataset | test items | {metric} | 95% CI | precision | recall | ROC-AUC | avg precision | decision threshold |"
EXTRA_METRICS = ("precision", "recall", "roc_auc", "avg_precision")


def _suite_result(f: CardFacts, suite: str, r: dict) -> dict:
    ds_name, ds_type = suite_dataset(f.base, suite)
    mm = r["metrics"]
    metric = f.base.objective.metric
    metrics = [
        {
            "type": metric,
            "value": round(r["objective"], 4),
            "name": f"{metric} ({f.base.decision.kind} decision, calibrated)",
        }
    ]
    metrics += [{"type": k, "value": round(mm[k], 4), "name": k} for k in EXTRA_METRICS if mm.get(k) is not None]
    return {
        "task": {"type": f.tag, "name": TASK_NAME.get(f.tag, f.tag)},
        "dataset": {"name": ds_name, "type": ds_type, "split": "test" if suite != "builtin" else "byosm-heldout"},
        "metrics": metrics,
    }


def front_matter(f: CardFacts) -> dict:
    results = [_suite_result(f, suite, r) for suite, r in f.results.items()]
    tags = ["byosm", "onnx", "system-1", f.spec.modality] + (["jepa", "lejepa"] if f.jepa else []) + f.encoders
    return {
        "license": f.license,
        "library_name": "onnx",
        "pipeline_tag": f.tag,
        "tags": tags,
        "datasets": list(f.datasets),
        "model-index": [{"name": f.repo_name, "results": results}],
    }


def _comparison_rows(f: CardFacts) -> list[tuple[str, dict]]:
    line_ids = {st["exp"] for st in f.lineage}
    keep = [
        (pid, by)
        for pid, by in f.board.items()
        if pid == f.exp or pid in line_ids or next(iter(by.values()))["kind"] != "experiment"
    ]
    return sorted(keep, key=lambda kv: -(kv[1].get(f.prim, {}).get("objective") or -1))


def _comparison_head(f: CardFacts, suites: list[str]) -> list[str]:
    metric = f.base.objective.metric
    objective_cols = " | ".join(f"{s} {metric}" + (" (primary)" if s == f.prim else "") for s in suites)
    auc_cols = " | ".join(f"{s} ROC-AUC" for s in suites)
    return ["| model | params | " + objective_cols + " | " + auc_cols + " |", "|" + "---|" * (2 + 2 * len(suites))]


def _comparison_row(f: CardFacts, pid: str, by: dict, suites: list[str]) -> str:
    first = next(iter(by.values()))
    kind = first["kind"]
    if pid == f.exp:
        label = f"**{pid} (this model)**"
    else:
        label = f"{pid} (BYOSM experiment)" if kind == "experiment" else f"{pid} (baseline)"
    params = first.get("params")
    cells = [label, f"{params / 1e6:.2f}M" if params else "–"]
    cells += [
        f"**{fmt(by[s]['objective'])}**" if pid == f.exp and s in by else fmt(by.get(s, {}).get("objective"))
        for s in suites
    ]
    cells += [fmt(by.get(s, {}).get("metrics", {}).get("roc_auc"), 3) for s in suites]
    return "| " + " | ".join(cells) + " |"


def comparison_table(f: CardFacts) -> str:
    suites = [e.name for e in f.base.evals]
    lines = _comparison_head(f, suites)
    lines += [_comparison_row(f, pid, by, suites) for pid, by in _comparison_rows(f)]
    return "\n".join(lines)


def _detail_row(f: CardFacts, suite: str, r: dict) -> str:
    mm = r["metrics"]
    ci = mm.get("ci95")
    ci_s = f"{ci['lo']:.3f}-{ci['hi']:.3f}" if ci else ""
    cells = [
        suite,
        suite_dataset(f.base, suite)[0],
        str(r["n_test"]),
        fmt(r["objective"]),
        ci_s,
        fmt(mm.get("precision")),
        fmt(mm.get("recall")),
        fmt(mm.get("roc_auc")),
        fmt(mm.get("avg_precision")),
        fmt(mm.get("threshold"), 3),
    ]
    return "| " + " | ".join(cells) + " |"


def detail_table(f: CardFacts) -> str:
    det = [DETAIL_HEAD.format(metric=f.base.objective.metric), "|---|---|---|---|---|---|---|---|---|---|"]
    det += [_detail_row(f, suite, r) for suite, r in f.results.items()]
    return "\n".join(det)
