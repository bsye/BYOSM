from __future__ import annotations

import time
from typing import Any

from ..spec.backbones import BACKBONES
from .standing import standing
from .state import Session

UNREACHABLE = "unreachable"
BIGGER = "bigger-backbone"
ENGLISH = "en"


def _speaks(entry: dict[str, Any], languages: list[str]) -> bool:
    return languages == [ENGLISH] or not entry["langs"].startswith(ENGLISH)


def bigger_backbone(s: Session) -> str | None:
    spec = s.p.exp_spec(str(s.p.promoted))
    options = BACKBONES.get(spec.modality, [])
    size = next((b["params_m"] for b in options if b["id"] == spec.backbone), 0)
    tried = {s.p.exp_spec(e).backbone for e in s.p.attempts()}
    larger = [b for b in options if b["params_m"] > size and b["id"] not in tried and _speaks(b, spec.languages)]
    return min(larger, key=lambda b: b["params_m"])["id"] if larger else None


def _try_bigger(s: Session, backbone: str) -> None:
    from ..improve import apply_ops
    from ..pipeline import build, next_exp

    parent = str(s.p.promoted)
    ops = [{"op": "set", "path": "backbone", "value": backbone}]
    why = f"every smaller fix stopped helping, so try the next larger backbone, {backbone}"
    exp = next_exp(s.p)
    proposal = {"parent": parent, "applied": [{"id": BIGGER, "ops": ops, "why": why}]}
    s.p.save_experiment(exp, spec=apply_ops(s.p.exp_spec(parent), ops), proposal=proposal)
    s.log("escalate", exp, why)
    try:
        build(s.name, exp, s.echo, data_seconds=s.data_seconds())
    except Exception as e:
        s.log("escalate", exp, f"{BIGGER} failed: {type(e).__name__}: {str(e)[:200]}")


def escalate(s: Session, reason: str) -> str:
    backbone = bigger_backbone(s)
    if backbone and time.time() + s.mean_duration() <= s.deadline:
        _try_bigger(s, backbone)
        st = standing(s.name, s.margin)
        if st["won"]:
            return f"won after escalating to {backbone}"
    if backbone:
        return reason
    st = standing(s.name, s.margin)
    tried = sorted({s.p.exp_spec(e).backbone for e in s.p.attempts()})
    note = {"at": time.time(), "model": st["exp"], "ours": st["ours"], "target": st["target"], "tried": tried}
    s.p.save_report(UNREACHABLE, {**note, "after": reason})
    s.log("escalate", None, f"unreachable at this budget: every larger backbone was tried ({', '.join(tried)})")
    return f"unreachable at this budget after {reason}"
