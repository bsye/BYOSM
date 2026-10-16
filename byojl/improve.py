"""Diagnose the board and propose the next experiment.

Proposals are small, explicit edits to the spec (ops on spec.yaml paths), each with the evidence that motivated it:
    {"id": "real-data-ava", "why": "...", "ops": [{"op": "append", "path": "data.hf_sources", "value": {...}}]}
Built-in rules always run (no LLM needed); if an LLM is configured (role `analyze`) it adds its own proposals.
`byojl improve vad --apply real-data-ava,utterance-labels` writes experiments/<next>/spec.yaml; `--run` also builds it.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

from .config import Project
from .spec import TaskSpec

ALLOWED_PATHS = [
    "data.hf_sources", "data.hf_queries", "data.train_size", "data.audio.snr_db", "data.audio.fill_ms", "data.audio.max_events",
    "data.audio.label_by_energy", "data.audio.clip_seconds", "data.synthetic", "data.per_call", "data.judge", "hard_cases",
    "diversity_axes", "backbone", "train.epochs", "train.pretrain_steps", "train.pretrain_lambda", "train.lr_backbone",
    "train.lr_head", "train.batch_size", "train.freeze_backbone", "train.refine_rounds", "train.refine_examples",
]


def diagnose(name: str) -> dict[str, Any]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    board = json.loads((p.root / "board.json").read_text()) if (p.root / "board.json").exists() else {}
    exp = p.promoted or next((k for k, v in board.items() if next(iter(v.values()))["kind"] == "experiment"), None)
    if not exp or exp not in board:
        raise RuntimeError("no scored experiment yet - run `byojl build` / `byojl eval` first")
    spec = TaskSpec.load(p.exp_dir(exp) / "spec.yaml")
    meta = json.loads((p.exp_dir(exp) / "meta.json").read_text())
    suites = {}
    for s, r in board[exp].items():
        bl = {k: v[s]["objective"] for k, v in board.items() if s in v and v[s]["kind"] == "baseline"}
        m = r["metrics"]
        suites[s] = {"objective": r["objective"], "precision": m.get("precision"), "recall": m.get("recall"), "threshold": m.get("threshold"),
                     "roc_auc": m.get("roc_auc"), "best_baseline": max(bl.items(), key=lambda kv: kv[1]) if bl else None}
    hist = meta.get("history", [])
    return {"experiment": exp, "spec": spec, "base": base, "suites": suites, "dev": meta["dev"].get("objective"),
            "still_improving": len(hist) >= 3 and hist[-1]["dev_objective"] >= max(h["dev_objective"] for h in hist[:-1]) - 1e-4,
            "epochs_run": len(hist), "epochs_cfg": spec.train.epochs, "board": board}


def rule_proposals(dg: dict[str, Any]) -> list[dict]:
    spec: TaskSpec = dg["spec"]
    base: TaskSpec = dg["base"]
    props: list[dict] = []
    trained_on = {(s.dataset, s.format) for s in spec.data.hf_sources if s.role == "labeled" and s.partition in ("train", "all")}
    builtin = dg["suites"].get("builtin", {}).get("objective")
    for suite in base.evals:
        r = dg["suites"].get(suite.name)
        if not r or suite.kind != "hf" or not suite.source:
            continue
        gap = (builtin - r["objective"]) if builtin is not None else None
        bb = r["best_baseline"]
        losing = bb and bb[1] > r["objective"]
        # 1. In-domain real data from the same source, disjoint partition.
        if (suite.source.dataset, suite.source.format) not in trained_on and (losing or (gap or 0) > 0.05):
            src = suite.source.model_copy(update={"role": "labeled", "partition": "train", "max_files": 60, "max_rows": 6000})
            props.append({"id": f"real-data-{suite.name}", "ops": [{"op": "append", "path": "data.hf_sources", "value": src.model_dump(mode="json")}],
                          "why": f"{suite.name}: objective {r['objective']:.3f}" + (f" vs {bb[0]} {bb[1]:.3f}" if bb else "") +
                                 (f", {gap:.3f} below builtin" if gap else "") + ". The model has never seen this domain; train on the source's "
                                 "train partition (hash-disjoint from the suite's calib/test files)."})
        # 2. Label convention mismatch (frame tasks): calibration pushed the threshold to an extreme.
        thr = r.get("threshold")
        if spec.task_type == "frame" and thr is not None and (thr <= 0.2 or thr >= 0.8) and spec.modality == "audio" and spec.data.audio.fill_ms < 300:
            props.append({"id": "utterance-labels", "ops": [{"op": "set", "path": "data.audio.fill_ms", "value": 300}],
                          "why": f"{suite.name}: calibration had to move the decision threshold to {thr:.2f} (P={r['precision']:.3f}, "
                                 f"R={r['recall']:.3f}) -> the suite's label convention differs from training's. Training labels only "
                                 f"voiced frames (gaps > {spec.data.audio.fill_ms}ms are negative); segment suites usually label whole "
                                 "utterances. Fill gaps up to 300ms in synthetic labels."})
    # 3. Under-trained.
    if dg["still_improving"] and dg["epochs_run"] >= dg["epochs_cfg"]:
        props.append({"id": "train-longer", "ops": [{"op": "set", "path": "train.epochs", "value": int(spec.train.epochs * 1.6)}],
                      "why": f"dev objective was still rising at the last epoch ({dg['epochs_run']} epochs)."})
    # 4. Text: error-driven data.
    if spec.modality == "text" and spec.data.synthetic and (dg["dev"] or 0) < spec.objective.target:
        props.append({"id": "refine-more", "ops": [{"op": "set", "path": "train.refine_rounds", "value": spec.train.refine_rounds + 2}],
                      "why": f"dev {dg['dev']:.3f} < target {spec.objective.target}; run more analyze->generate->retrain rounds."})
    return props


LLM_SYSTEM = """You are the research lead improving a tiny single-purpose model. Given its spec, the board (our experiment \
vs baselines on several frozen eval suites, with precision/recall/threshold), propose at most 4 concrete next experiments. \
Each proposal is a list of edits to the spec at the allowed paths. Prefer data fixes (coverage of the failing domain, label \
conventions, hard negatives) over model size. Never propose training on an eval suite's calib/test data. Be specific: \
exact Hugging Face dataset ids you are confident exist, exact values."""


def llm_proposals(dg: dict[str, Any], log: Callable[[str], None]) -> list[dict]:
    from .llm import get_llm

    schema = {"type": "object", "properties": {"proposals": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "why": {"type": "string"},
        "ops": {"type": "array", "items": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["set", "append"]}, "path": {"type": "string", "enum": ALLOWED_PATHS}, "value_json": {"type": "string"}}}}}}}}}
    board = {k: {s: {"objective": round(r["objective"], 4), **{m: r["metrics"].get(m) for m in ("precision", "recall", "threshold", "roc_auc")}} for s, r in v.items()} for k, v in dg["board"].items()}
    prompt = f"Spec:\n{dg['spec'].model_dump_json(indent=1)}\n\nBoard:\n{json.dumps(board, indent=1)}\n\nAllowed paths: {ALLOWED_PATHS}"
    try:
        out = get_llm("analyze").json(LLM_SYSTEM, prompt, schema)
    except Exception as e:  # noqa: BLE001
        log(f"  (LLM proposals skipped: {type(e).__name__}: {str(e)[:100]})")
        return []
    props = []
    for pr in out.get("proposals", []):
        try:
            ops = [{"op": o["op"], "path": o["path"], "value": json.loads(o["value_json"])} for o in pr["ops"] if o["path"] in ALLOWED_PATHS]
        except (json.JSONDecodeError, KeyError):
            continue
        if ops:
            props.append({"id": "llm-" + pr["id"], "why": pr["why"], "ops": ops})
    return props


def apply_ops(spec: TaskSpec, ops: list[dict]) -> TaskSpec:
    d = copy.deepcopy(spec.model_dump(mode="json"))
    for o in ops:
        *head, last = o["path"].split(".")
        cur = d
        for k in head:
            cur = cur[k]
        if o["op"] == "append":
            cur.setdefault(last, []).append(o["value"])
        else:
            cur[last] = o["value"]
    return TaskSpec.model_validate(d)


def propose(name: str, use_llm: bool = True, log: Callable[[str], None] = print) -> tuple[dict, list[dict]]:
    dg = diagnose(name)
    props = rule_proposals(dg) + (llm_proposals(dg, log) if use_llm else [])
    p = Project(name)
    (p.root / "proposals.json").write_text(json.dumps({"from": dg["experiment"], "proposals": props}, indent=2))
    return dg, props


def apply(name: str, ids: list[str], exp: str | None = None) -> str:
    from .pipeline import next_exp

    p = Project(name)
    saved = json.loads((p.root / "proposals.json").read_text())
    chosen = [pr for pr in saved["proposals"] if pr["id"] in ids]
    if len(chosen) != len(ids):
        raise KeyError(f"unknown proposal ids: {set(ids) - {pr['id'] for pr in chosen}}")
    parent = saved["from"]
    spec = apply_ops(TaskSpec.load(p.exp_dir(parent) / "spec.yaml"), [o for pr in chosen for o in pr["ops"]])
    exp = exp or next_exp(p)
    ed = p.exp_dir(exp)
    ed.mkdir(parents=True, exist_ok=True)
    spec.save(ed / "spec.yaml")
    (ed / "proposal.json").write_text(json.dumps({"parent": parent, "applied": chosen}, indent=2))
    return exp
