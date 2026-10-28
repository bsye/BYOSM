from __future__ import annotations

from typing import Any

from ..spec.sections import EMBEDDING_HEAD, FINETUNE, SETFIT
from .registry import gap, refine_hurt, rule

FEW_SHOT_PER_LABEL = 30

DOMAIN_ENCODERS: dict[str, dict[str, Any]] = {
    "document.image_backbone": {
        "good": ["hf:microsoft/dit-base", "hf:microsoft/dit-large", "tiny-cnn"],
        "propose": "hf:microsoft/dit-base",
        "why": "self-supervised (masked image modelling) on 42M unlabelled document scans; never saw RVL-CDIP labels",
    },
}


@rule
def regularize(dg: dict) -> list[dict]:
    spec, curve, losses, best = dg["spec"], dg["curve"], dg["losses"], dg["best_epoch"]
    if len(curve) < 4 or best >= len(curve) - 2 or not losses or losses[-1] > 0.75 * losses[best]:
        return []
    ops = [{"op": "set", "path": "train.lr_backbone", "value": spec.train.lr_backbone / 2}]
    extra = ""
    if spec.modality == "document" and spec.document.route == "fused" and spec.document.modality_dropout < 0.4:
        ops.append(
            {"op": "set", "path": "document.modality_dropout", "value": round(spec.document.modality_dropout + 0.15, 2)}
        )
        extra = " and drop a view more often"
    why = (
        f"dev peaked at epoch {best + 1} ({curve[best]:.3f}) and never improved after, "
        f"while training loss fell {losses[best]:.3f} -> {losses[-1]:.3f}: "
        f"overfitting. Halve the encoder LR ({spec.train.lr_backbone:g} -> {spec.train.lr_backbone / 2:g}){extra}."
    )
    return [{"id": "regularize", "ops": ops, "why": why}]


@rule
def domain_encoder(dg: dict) -> list[dict]:
    spec, out = dg["spec"], []
    lag, bb = gap(dg, dg["primary"])
    if lag <= 0.02:
        return out
    for path, info in DOMAIN_ENCODERS.items():
        section, key = path.split(".")
        cur = getattr(getattr(spec, section), key, None)
        if spec.modality != section or cur in info["good"]:
            continue
        if section == "document" and spec.document.route == "text":
            continue
        why = f"{dg['primary']}: {lag:.3f} behind {bb[0]}. `{cur}` was pretrained on a different domain; {info['propose']} is {info['why']}."
        out.append({"id": "domain-encoder", "ops": [{"op": "set", "path": path, "value": info["propose"]}], "why": why})
    return out


@rule
def train_longer(dg: dict) -> list[dict]:
    if not dg["late_best"]:
        return []
    spec = dg["spec"]
    ops = [{"op": "set", "path": "train.epochs", "value": int(round(spec.train.epochs * 1.6))}]
    recent = ", ".join(f"{c:.2f}" for c in dg["curve"][-4:])
    why = f"best dev epoch was {dg['best_epoch'] + 1} of {dg['epochs_run']} (curve {recent}): still converging."
    return [{"id": "train-longer", "ops": ops, "why": why}]


@rule
def label_convention(dg: dict) -> list[dict]:
    spec, base = dg["spec"], dg["base"]
    if spec.task_type != "frame" or spec.modality != "audio" or spec.data.audio.fill_ms >= 300:
        return []
    for suite in base.evals:
        r = dg["suites"].get(suite.name)
        thr = r and r.get("threshold")
        if suite.kind == "hf" and thr is not None and (thr <= 0.2 or thr >= 0.8):
            why = (
                f"{suite.name}: calibration moved the threshold to {thr:.2f} -> its label convention differs from training's "
                f"(gaps > {spec.data.audio.fill_ms}ms are negative in training; segment suites usually label whole utterances)."
            )
            return [
                {
                    "id": "utterance-labels",
                    "ops": [{"op": "set", "path": "data.audio.fill_ms", "value": 300}],
                    "why": why,
                }
            ]
    return []


@rule
def confusable_pairs(dg: dict) -> list[dict]:
    spec = dg["spec"]
    conf = dg["suites"].get(dg["primary"], {}).get("confusions") or []
    if not conf or not spec.data.synthetic or refine_hurt(dg):
        return []
    cases = [f"{g} that looks like {p} (it is {g})" for g, p, _ in conf[:4]]
    ops = [
        {"op": "set", "path": "hard_cases", "value": spec.hard_cases + cases},
        {"op": "set", "path": "train.refine_rounds", "value": spec.train.refine_rounds + 1},
    ]
    pairs = ", ".join(f"{g}->{p} ({c})" for g, p, c in conf[:4])
    return [
        {
            "id": "hard-pairs",
            "ops": ops,
            "why": "most confused: " + pairs + "; write targeted examples for these boundaries.",
        }
    ]


@rule
def refine_more(dg: dict) -> list[dict]:
    spec = dg["spec"]
    if (
        spec.modality == "text"
        and spec.data.synthetic
        and (dg["dev"] or 0) < spec.objective.target
        and not refine_hurt(dg)
    ):
        ops = [{"op": "set", "path": "train.refine_rounds", "value": spec.train.refine_rounds + 2}]
        why = f"dev {dg['dev']:.3f} < target {spec.objective.target}; run more analyze->generate->retrain rounds."
        return [{"id": "refine-more", "ops": ops, "why": why}]
    return []


@rule
def few_shot(dg: dict) -> list[dict]:
    spec = dg["spec"]
    per_label = (dg.get("n_train") or 0) / max(1, len(spec.labels))
    if spec.modality != "text" or spec.task_type != "classify" or spec.train.method != FINETUNE:
        return []
    if per_label >= FEW_SHOT_PER_LABEL:
        return []
    why = f"{per_label:.0f} training rows per label: below {FEW_SHOT_PER_LABEL}, a sentence-embedding model with "
    return [
        {
            "id": "setfit",
            "ops": [{"op": "set", "path": "train.method", "value": SETFIT}],
            "why": why + "contrastive tuning (SetFit) beats fine-tuning the encoder.",
        },
        {
            "id": "embedding-head",
            "ops": [{"op": "set", "path": "train.method", "value": EMBEDDING_HEAD}],
            "why": why + "a logistic head is a floor that trains in seconds.",
        },
    ]
