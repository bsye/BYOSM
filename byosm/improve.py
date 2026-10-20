"""Diagnose the board and propose the next experiment.

Proposals are small, explicit edits to the spec (ops on spec.yaml paths), each with the evidence that motivated it:
    {"id": "more-data", "why": "...", "ops": [{"op": "set", "path": "data.hf_sources.0.per_label", "value": 800}]}
Paths walk dicts by key and lists by index. Built-in rules always run (no LLM needed); if an LLM is configured
(role `analyze`) it adds its own proposals. `byosm improve <name> --apply a,b --run` builds the next experiment.

Each rule is a function (diagnosis) -> list[proposal]; add one with @rule.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any, Callable

from .config import Project
from .spec import TaskSpec

ALLOWED = [
    r"data\.hf_sources", r"data\.hf_sources\.\d+\.(per_label|max_rows|max_files|partition)", r"data\.hf_queries", r"data\.(train_size|dev_size|test_size)",
    r"data\.audio\.(snr_db|fill_ms|max_events|label_by_energy|clip_seconds)", r"data\.(synthetic|per_call|judge)", r"hard_cases", r"diversity_axes",
    r"backbone", r"train\.(epochs|pretrain_steps|pretrain_lambda|lr_backbone|lr_head|batch_size|freeze_backbone|refine_rounds|refine_examples|pool"
    r"|layer_decay_floor|probe_epochs|ema|precision|seed)",
    r"document\.(route|text_backbone|image_backbone|max_pages|max_len|image_size|ocr|dim|modality_dropout)",
]


def allowed(path: str) -> bool:
    return any(re.fullmatch(a, path) for a in ALLOWED)


# Encoders pretrained on the right domain. Swapping a general encoder for one of these is often the largest single gain.
DOMAIN_ENCODERS: dict[str, dict[str, Any]] = {
    "document.image_backbone": {
        "good": ["hf:microsoft/dit-base", "hf:microsoft/dit-large", "tiny-cnn"],
        "propose": "hf:microsoft/dit-base",
        "why": "self-supervised (masked image modelling) on 42M unlabelled document scans; never saw RVL-CDIP labels",
    },
}


# --------------------------------------------------------------------------------------------- diagnosis


def _confusions(p: Project, exp: str, suite: str, spec: TaskSpec, k: int = 6) -> list[tuple[str, str, int]]:
    """Most frequent (gold, predicted) pairs: from the suite's confusion matrix, else from dev errors."""
    f = p.exp_dir(exp) / "results" / f"{suite}.json"
    cm = json.loads(f.read_text())["metrics"].get("confusion") if f.exists() else None
    names = spec.label_names
    if cm:
        pairs = [(names[i], names[j], c) for i, row in enumerate(cm) for j, c in enumerate(row) if i != j and c]
    else:
        ef = p.exp_dir(exp) / "dev_errors.json"
        errs = json.loads(ef.read_text()) if ef.exists() else []
        pairs = [(g, pr, c) for (g, pr), c in Counter((e["gold"], e["pred"]) for e in errs if e.get("gold") != "frames").items()]
    return sorted(pairs, key=lambda t: -t[2])[:k]


def diagnose(name: str) -> dict[str, Any]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    board = p.board()
    exp = p.promoted or next((k for k, v in board.items() if next(iter(v.values()))["kind"] == "experiment"), None)
    if not exp or exp not in board:
        raise RuntimeError("no scored experiment yet - run `byosm build` / `byosm eval` first")
    spec = TaskSpec.load(p.exp_dir(exp) / "spec.yaml")
    meta = json.loads((p.exp_dir(exp) / "meta.json").read_text())
    binary = spec.positive_label is not None or len(spec.labels) == 2
    suites = {}
    for s, r in board[exp].items():
        bl = {k: v[s]["objective"] for k, v in board.items() if s in v and v[s]["kind"] == "baseline"}
        m = r["metrics"]
        weakest = sorted(((n, v["f1"]) for n, v in m.get("per_label", {}).items()), key=lambda t: t[1])[:3]
        suites[s] = {"objective": r["objective"], "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"), "roc_auc": m.get("roc_auc"),
                     "precision": m.get("precision") if binary else None, "recall": m.get("recall") if binary else None, "threshold": m.get("threshold"),
                     "best_baseline": max(bl.items(), key=lambda kv: kv[1]) if bl else None, "weakest": weakest,
                     "confusions": _confusions(p, exp, s, spec)}
    hist = meta.get("history", [])
    curve = [h["dev_objective"] for h in hist]
    losses = [h["loss"] for h in hist]
    best_ep = meta.get("best_epoch", len(curve) - 1)
    return {"experiment": exp, "spec": spec, "base": base, "suites": suites, "dev": meta["dev"].get("objective"), "binary": binary,
            "curve": curve, "losses": losses, "best_epoch": best_ep, "train_seconds": meta.get("train_seconds"), "epochs_run": len(curve), "epochs_cfg": spec.train.epochs, "n_train": meta.get("n_train"),
            # converging late = the best epoch is in the last quarter of a run that wasn't early-stopped
            "late_best": len(curve) >= 4 and len(curve) >= spec.train.epochs and best_ep >= int(0.75 * len(curve)),
            "board": board, "primary": base.primary_suite()}


# --------------------------------------------------------------------------------------------- rules

RULES: list[Callable[[dict], list[dict]]] = []


def rule(fn: Callable[[dict], list[dict]]) -> Callable[[dict], list[dict]]:
    RULES.append(fn)
    return fn


def _gap(dg: dict, suite: str) -> tuple[float, tuple[str, float] | None]:
    r = dg["suites"].get(suite, {})
    bb = r.get("best_baseline")
    return ((bb[1] - r["objective"]) if bb else 0.0), bb


@rule
def real_data(dg: dict) -> list[dict]:
    """A real-world suite lags, and the model never trained on that source: use its (hash-disjoint) train partition."""
    spec, base, out = dg["spec"], dg["base"], []
    trained = {s.dataset for s in spec.data.hf_sources if s.role == "labeled"}
    builtin = dg["suites"].get("builtin", {}).get("objective")
    for suite in base.evals:
        r = dg["suites"].get(suite.name)
        if not r or suite.kind != "hf" or not suite.source or suite.source.dataset in trained:
            continue
        lag, bb = _gap(dg, suite.name)
        dgap = (builtin - r["objective"]) if builtin is not None else 0
        if lag > 0 or dgap > 0.05:
            upd = {"role": "labeled", "partition": "train", "max_files": 60, "max_rows": 6000} if suite.source.format == "segment_files" else {"role": "labeled", "split": "train" if suite.source.split != "train" else suite.source.split}
            src = suite.source.model_copy(update=upd)
            out.append({"id": f"real-data-{suite.name}", "ops": [{"op": "append", "path": "data.hf_sources", "value": src.model_dump(mode="json")}],
                        "why": f"{suite.name}: {r['objective']:.3f}" + (f" vs {bb[0]} {bb[1]:.3f}" if bb else "") + (f", {dgap:.3f} below builtin" if dgap > 0 else "")
                               + ". The model never trained on this source; use its train partition (disjoint from the suite's calib/test)."})
    return out


@rule
def more_data(dg: dict) -> list[dict]:
    """Losing to a baseline (or below target) while the labelled source has more to give: take ~4x more."""
    spec, out = dg["spec"], []
    lag, bb = _gap(dg, dg["primary"])
    below = (dg["suites"].get(dg["primary"], {}).get("objective") or 0) < spec.objective.target
    if not (lag > 0.02 or below):
        return out
    obj = dg["suites"][dg["primary"]]["objective"]
    ev = f"{dg['primary']}: {obj:.3f}, " + (f"{lag:.3f} behind {bb[0]} ({bb[1]:.3f})" if lag > 0.02 else f"below target {spec.objective.target}") + f", with {dg['n_train']} training examples"
    curve = dg["curve"]
    if len(curve) >= 2:
        tail = curve[-max(2, len(curve) // 3):]
        if dg["late_best"]:
            ev += f"; dev was still rising at the end ({curve[0]:.2f} -> {max(curve):.2f}), consistent with being data-limited"
        elif max(tail) - min(tail) < 0.01:
            ev += f"; dev plateaued at {max(curve):.2f}, so more of the same data may help less than new data or a new recipe"
        else:
            ev += f"; dev reached {max(curve):.2f}"
    for i, s in enumerate(spec.data.hf_sources):
        if s.role != "labeled":
            continue
        if s.sampling == "balanced":
            out += _more_balanced(dg, i, s, ev)
        elif s.format == "segment_files":
            out.append({"id": "more-data", "ops": [{"op": "set", "path": f"data.hf_sources.{i}.max_files", "value": s.max_files * 2},
                                                   {"op": "set", "path": f"data.hf_sources.{i}.max_rows", "value": s.max_rows * 2}],
                        "why": ev + f". Take 2x more files/clips from {s.dataset}."})
        else:
            out.append({"id": "more-data", "ops": [{"op": "set", "path": f"data.hf_sources.{i}.max_rows", "value": s.max_rows * 4}],
                        "why": ev + f". Read 4x more rows from {s.dataset}."})
        break
    if not out and spec.data.synthetic:
        out.append({"id": "more-data", "ops": [{"op": "set", "path": "data.train_size", "value": spec.data.train_size * 2}], "why": ev + ". Generate 2x more training data."})
    return out


def _more_balanced(dg: dict, i: int, s: Any, ev: str) -> list[dict]:
    """4x more rows per label - capped at what the split really has; when it is exhausted, add the dataset's largest split
    that no eval suite reads (e.g. RVL-CDIP train, 320k rows, while training on val)."""
    from .data import hf_source

    spec, base = dg["spec"], dg["base"]
    want = s.per_label * 4
    try:
        have = min(hf_source.availability(s, spec).values() or [0])
    except Exception:  # noqa: BLE001
        have = None
    if have is None or want <= have:
        return [{"id": "more-data", "ops": [{"op": "set", "path": f"data.hf_sources.{i}.per_label", "value": want}],
                 "why": ev + f". {s.dataset} ({s.split}) has more rows per label: {s.per_label} -> {want}."}]
    if have >= s.per_label * 1.5:
        return [{"id": "more-data", "ops": [{"op": "set", "path": f"data.hf_sources.{i}.per_label", "value": have}],
                 "why": ev + f". {s.dataset} ({s.split}) has {have} rows for its rarest label: {s.per_label} -> {have} per label (all of it)."}]
    used = {e.source.split for e in base.evals if e.source and e.source.dataset == s.dataset} | {x.split for x in spec.data.hf_sources if x.dataset == s.dataset}
    sizes = hf_source.split_sizes(s.dataset, s.config)
    alt = max((k for k in sizes if k not in used), key=lambda k: sizes[k], default=None)
    if not alt or sizes[alt] // len(spec.labels) <= s.per_label // 2:
        return []
    per = min(want, sizes[alt] // len(spec.labels))
    return [{"id": "new-split", "ops": [{"op": "append", "path": "data.hf_sources", "value": s.model_copy(update={"split": alt, "per_label": per}).model_dump(mode="json")}],
             "why": ev + f". {s.split} is used up ({have} per label, {s.per_label} taken); `{alt}` has {sizes[alt]} rows that no eval suite reads: add {per} per label from it."}]


@rule
def regularize(dg: dict) -> list[dict]:
    """Overfitting: dev peaked early and stopped improving while the training loss kept falling."""
    spec, curve, losses, best = dg["spec"], dg["curve"], dg["losses"], dg["best_epoch"]
    if len(curve) < 4 or best >= len(curve) - 2 or not losses or losses[-1] > 0.75 * losses[best]:
        return []
    ops = [{"op": "set", "path": "train.lr_backbone", "value": spec.train.lr_backbone / 2}]
    extra = ""
    if spec.modality == "document" and spec.document.route == "fused" and spec.document.modality_dropout < 0.4:
        ops.append({"op": "set", "path": "document.modality_dropout", "value": round(spec.document.modality_dropout + 0.15, 2)})
        extra = " and drop a view more often"
    return [{"id": "regularize", "ops": ops,
             "why": f"dev peaked at epoch {best + 1} ({curve[best]:.3f}) and never improved after, while training loss fell {losses[best]:.3f} -> {losses[-1]:.3f}: "
                    f"overfitting. Halve the encoder LR ({spec.train.lr_backbone:g} -> {spec.train.lr_backbone / 2:g}){extra}."}]


@rule
def domain_encoder(dg: dict) -> list[dict]:
    """An encoder pretrained on the wrong domain (e.g. photos for documents) while losing to a baseline."""
    spec, out = dg["spec"], []
    lag, bb = _gap(dg, dg["primary"])
    if lag <= 0.02:
        return out
    for path, info in DOMAIN_ENCODERS.items():
        section, key = path.split(".")
        cur = getattr(getattr(spec, section), key, None)
        if spec.modality != section or cur in info["good"]:
            continue
        if section == "document" and spec.document.route == "text":
            continue
        out.append({"id": "domain-encoder", "ops": [{"op": "set", "path": path, "value": info["propose"]}],
                    "why": f"{dg['primary']}: {lag:.3f} behind {bb[0]}. `{cur}` was pretrained on a different domain; {info['propose']} is {info['why']}."})
    return out


@rule
def train_longer(dg: dict) -> list[dict]:
    if not dg["late_best"]:
        return []
    spec = dg["spec"]
    return [{"id": "train-longer", "ops": [{"op": "set", "path": "train.epochs", "value": int(round(spec.train.epochs * 1.6))}],
             "why": f"best dev epoch was {dg['best_epoch'] + 1} of {dg['epochs_run']} (curve {', '.join(f'{c:.2f}' for c in dg['curve'][-4:])}): still converging."}]


@rule
def label_convention(dg: dict) -> list[dict]:
    """Frame audio: calibration pushed the threshold to an extreme -> the suite labels differently from training."""
    spec, base, out = dg["spec"], dg["base"], []
    if spec.task_type != "frame" or spec.modality != "audio" or spec.data.audio.fill_ms >= 300:
        return out
    for suite in base.evals:
        r = dg["suites"].get(suite.name)
        thr = r and r.get("threshold")
        if suite.kind == "hf" and thr is not None and (thr <= 0.2 or thr >= 0.8):
            out.append({"id": "utterance-labels", "ops": [{"op": "set", "path": "data.audio.fill_ms", "value": 300}],
                        "why": f"{suite.name}: calibration moved the threshold to {thr:.2f} -> its label convention differs from training's "
                               f"(gaps > {spec.data.audio.fill_ms}ms are negative in training; segment suites usually label whole utterances)."})
            break
    return out


@rule
def confusable_pairs(dg: dict) -> list[dict]:
    """Systematic confusions. For LLM-written data: turn them into hard cases + an extra refine round."""
    spec = dg["spec"]
    conf = dg["suites"].get(dg["primary"], {}).get("confusions") or []
    if not conf or not spec.data.synthetic:
        return []
    cases = [f"{g} that looks like {p} (it is {g})" for g, p, _ in conf[:4]]
    return [{"id": "hard-pairs", "ops": [{"op": "set", "path": "hard_cases", "value": spec.hard_cases + cases},
                                         {"op": "set", "path": "train.refine_rounds", "value": spec.train.refine_rounds + 1}],
             "why": "most confused: " + ", ".join(f"{g}->{p} ({c})" for g, p, c in conf[:4]) + "; write targeted examples for these boundaries."}]


@rule
def refine_more(dg: dict) -> list[dict]:
    spec = dg["spec"]
    if spec.modality == "text" and spec.data.synthetic and (dg["dev"] or 0) < spec.objective.target:
        return [{"id": "refine-more", "ops": [{"op": "set", "path": "train.refine_rounds", "value": spec.train.refine_rounds + 2}],
                 "why": f"dev {dg['dev']:.3f} < target {spec.objective.target}; run more analyze->generate->retrain rounds."}]
    return []


# Ingest cost per new labelled item (seconds), measured on an M5 Max: OCR ~0.55 s/page (RapidOCR, CPU-bound), media decode otherwise.
INGEST_S = {"document": 0.36,  # OCR, one engine per core (M5 Max, RVL-CDIP scans)
            "image": 0.02, "audio": 0.05, "text": 0.0}


def _items(spec: TaskSpec) -> int:
    n = 0
    for s in spec.data.hf_sources:
        if s.role == "labeled":
            n += s.per_label * len(spec.labels) if s.sampling == "balanced" and s.format == "rows" else s.max_rows
    return n + (spec.data.train_size if spec.data.synthetic else 0)


def estimate(dg: dict, pr: dict) -> dict[str, float]:
    """Rough cost of a proposal: ingest of the new data + training time scaled by data size and epochs."""
    old = dg["spec"]
    try:
        new = apply_ops(old, pr["ops"])
    except Exception:  # noqa: BLE001
        return {"minutes": float("nan"), "new_items": 0}
    a, b = max(1, _items(old)), _items(new)
    scale = (b / a) * (new.train.epochs / max(1, old.train.epochs))
    if new.backbone != old.backbone or new.document.image_backbone != old.document.image_backbone or new.document.text_backbone != old.document.text_backbone:
        scale *= 1.5  # a different encoder: unknown speed, assume somewhat slower
    ocr = new.modality == "document" and new.document.route != "visual"
    ingest = max(0, b - a) * (INGEST_S["document"] if ocr else INGEST_S["image"] if new.modality == "document" else INGEST_S.get(new.modality, 0))
    return {"minutes": round(((dg.get("train_seconds") or 0) * scale + ingest) / 60, 1), "new_items": max(0, b - a)}


def rule_proposals(dg: dict[str, Any]) -> list[dict]:
    seen, out = set(), []
    for r in RULES:
        for pr in r(dg):
            if pr["id"] not in seen:
                seen.add(pr["id"])
                out.append(pr)
    return out


# --------------------------------------------------------------------------------------------- LLM analyst

LLM_SYSTEM = """You are the research lead improving a tiny single-purpose model. Given its spec, the board (our experiment \
vs baselines on several frozen eval suites), the training curve and the most confused label pairs, propose at most 4 \
concrete next experiments. Each proposal is a list of edits to the spec at allowed paths (dotted; list items by index). \
Prefer data fixes (coverage of the failing domain, label conventions, hard negatives, more data) and domain-pretrained \
encoders over model size. Never propose training on an eval suite's calib/test data. Be specific: exact Hugging Face ids \
you are confident exist, exact values."""


def llm_proposals(dg: dict[str, Any], log: Callable[[str], None]) -> list[dict]:
    from .llm import get_llm

    schema = {"type": "object", "properties": {"proposals": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "why": {"type": "string"},
        "ops": {"type": "array", "items": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["set", "append"]}, "path": {"type": "string"}, "value_json": {"type": "string"}}}}}}}}}
    board = {k: {s: {"objective": round(r["objective"], 4), **{m: r["metrics"].get(m) for m in ("accuracy", "macro_f1", "precision", "recall", "threshold", "roc_auc")}} for s, r in v.items()} for k, v in dg["board"].items()}
    diag = {s: {k: v for k, v in r.items() if k in ("weakest", "confusions")} for s, r in dg["suites"].items()}
    prompt = (f"Spec:\n{dg['spec'].model_dump_json(indent=1)}\n\nBoard:\n{json.dumps(board, indent=1)}\n\nTraining curve (dev objective per epoch): {dg['curve']}\n"
              f"Diagnosis per suite: {json.dumps(diag)}\n\nAllowed path patterns (regex): {ALLOWED}")
    try:
        out = get_llm("analyze").json(LLM_SYSTEM, prompt, schema)
    except Exception as e:  # noqa: BLE001
        log(f"  (LLM proposals skipped: {type(e).__name__}: {str(e)[:100]})")
        return []
    props = []
    for pr in out.get("proposals", []):
        try:
            ops = [{"op": o["op"], "path": o["path"], "value": json.loads(o["value_json"])} for o in pr["ops"] if allowed(o["path"])]
        except (json.JSONDecodeError, KeyError):
            continue
        if ops:
            props.append({"id": "llm-" + pr["id"], "why": pr["why"], "ops": ops})
    return props


# --------------------------------------------------------------------------------------------- apply


def apply_ops(spec: TaskSpec, ops: list[dict]) -> TaskSpec:
    """Apply set/append ops. Paths are dotted; numeric parts index lists (data.hf_sources.0.per_label)."""
    d = copy.deepcopy(spec.model_dump(mode="json"))
    for o in ops:
        *head, last = o["path"].split(".")
        cur: Any = d
        for k in head:
            cur = cur[int(k)] if isinstance(cur, list) else cur[k]
        key: Any = int(last) if isinstance(cur, list) else last
        if o["op"] == "append":
            (cur[key] if isinstance(cur, list) else cur.setdefault(key, [])).append(o["value"])
        else:
            cur[key] = o["value"]
    return TaskSpec.model_validate(d)


def propose(name: str, use_llm: bool = True, log: Callable[[str], None] = print) -> tuple[dict, list[dict]]:
    dg = diagnose(name)
    props = rule_proposals(dg) + (llm_proposals(dg, log) if use_llm else [])
    for pr in props:
        pr["cost"] = estimate(dg, pr)
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
