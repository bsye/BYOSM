from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .paths import allowed, untrainable
from .techniques import eligible, tried_counts

LLM_SYSTEM = """You are the research lead improving a tiny single-purpose model. Given its spec, the board (our experiment \
vs baselines on several frozen eval suites), the training curve and the most confused label pairs, propose at most 4 \
concrete next experiments. Each proposal is a list of edits to the spec at the paths of one technique family from the \
catalogue (dotted; list items by index). Spread the proposals over different families: favour the families that fit \
the diagnosis and have been tried least, and try a family again only with a reason its earlier tries did not cover. \
Never propose training on an eval suite's calib/test data. Be specific: exact Hugging Face ids you are confident exist, \
exact values."""

_OP = {
    "op": {"type": "string", "enum": ["set", "append"]},
    "path": {"type": "string"},
    "value_json": {"type": "string"},
}
_PROPOSAL = {
    "id": {"type": "string"},
    "why": {"type": "string"},
    "ops": {"type": "array", "items": {"type": "object", "properties": _OP}},
}
_SCHEMA = {
    "type": "object",
    "properties": {"proposals": {"type": "array", "items": {"type": "object", "properties": _PROPOSAL}}},
}
_BOARD_METRICS = ("accuracy", "macro_f1", "precision", "recall", "threshold", "roc_auc")


def _board_view(board: dict) -> dict:
    return {
        k: {
            s: {"objective": round(r["objective"], 4), **{m: r["metrics"].get(m) for m in _BOARD_METRICS}}
            for s, r in v.items()
        }
        for k, v in board.items()
    }


def _prompt(dg: dict[str, Any]) -> str:
    board = _board_view(dg["board"])
    diag = {s: {k: v for k, v in r.items() if k in ("weakest", "confusions")} for s, r in dg["suites"].items()}
    return (
        f"Spec:\n{dg['spec'].model_dump_json(indent=1)}\n\nBoard:\n{json.dumps(board, indent=1)}\n\n"
        f"Training curve (dev objective per epoch): {dg['curve']}\n"
        f"Diagnosis per suite: {json.dumps(diag)}\n\n"
        f"Experiments tried so far, what each changed from its parent and how it scored on the primary suite "
        f"(calib picks, test is only reported). Do not repeat a change that did not help; combine and build on "
        f"what helped:\n{json.dumps(dg.get('history', []), indent=1)}\n\n"
        f"Technique catalogue (paths: dotted, | separates alternatives for one segment, * is a list index; "
        f"tried: experiments of this specialist in that family):\n{json.dumps(_catalogue(dg), indent=1)}"
    )


def _catalogue(dg: dict[str, Any]) -> list[dict[str, Any]]:
    tried = tried_counts(dg.get("history", []))
    return [
        {"family": t.id, "effect": t.effect, "paths": t.paths, "prior": t.prior, "tried": tried[t.id]}
        for t in eligible(dg["spec"])
    ]


def _parse(out: dict) -> list[dict]:
    props = []
    for pr in out.get("proposals", []):
        try:
            ops = [
                {"op": o["op"], "path": o["path"], "value": json.loads(o["value_json"])}
                for o in pr["ops"]
                if allowed(o["path"])
            ]
        except (json.JSONDecodeError, KeyError):
            continue
        if ops:
            props.append({"id": "llm-" + pr["id"], "why": pr["why"], "ops": ops})
    return props


def llm_proposals(dg: dict[str, Any], log: Callable[[str], None]) -> list[dict]:
    from ..llm import get_llm

    prompt = _prompt(dg)
    try:
        out = get_llm("analyze").json(LLM_SYSTEM, prompt, _SCHEMA)
    except Exception as e:
        log(f"  (LLM proposals skipped: {type(e).__name__}: {str(e)[:100]})")
        return []
    return [p for p in _parse(out) if _buildable(p, log)]


def _buildable(proposal: dict, log: Callable[[str], None]) -> bool:
    bad = untrainable(proposal["ops"])
    if bad:
        log(f"  (dropped {proposal['id']}: {', '.join(bad)} has no PyTorch weights to fine-tune)")
    return not bad
