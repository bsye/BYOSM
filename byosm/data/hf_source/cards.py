from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

CARD_CHARS = 280
DESCRIBED = 20
WORKERS = 8
CARD_FIELDS = ("task_categories", "license", "size_categories", "language")
UNKNOWN_LICENCE = "licence not stated on its card"


def card(hit: dict) -> dict[str, Any]:
    from huggingface_hub import HfApi

    from .hub import OPEN_PREFIX

    if hit["id"].startswith(OPEN_PREFIX):
        return hit

    try:
        info = HfApi().dataset_info(hit["id"])
    except Exception:
        return {**hit, "missing": True}
    data = info.card_data.to_dict() if info.card_data else {}
    about = " ".join((info.description or "").split())[:CARD_CHARS]
    return {
        **hit,
        **{k: data.get(k) for k in CARD_FIELDS},
        "title": data.get("pretty_name"),
        "about": about,
        "gated": bool(info.gated),
    }


def describe(hits: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(WORKERS) as pool:
        cards = list(pool.map(card, hits[:DESCRIBED]))
    return [c for c in cards if not c.get("missing")]


def _short(v: Any) -> str:
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v or "")


def listing(cards: list[dict]) -> str:
    rows = []
    for c in cards:
        facts = "; ".join(f"{k.replace('_', ' ')}: {_short(c.get(k))}" for k in CARD_FIELDS if c.get(k))
        gated = "; gated" if c.get("gated") else ""
        rows.append(
            f"- {c['id']} ({c.get('downloads') or 0} downloads; {facts}{gated}): {c.get('title') or ''} {c.get('about') or ''}".rstrip()
        )
    return "\n".join(rows)


def licence(dataset: str) -> str:
    from huggingface_hub import HfApi

    try:
        card = HfApi().dataset_info(dataset).card_data
    except Exception:
        return UNKNOWN_LICENCE
    return (card.get("license") if card else None) or UNKNOWN_LICENCE
