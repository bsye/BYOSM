from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import PUBLISH_LICENSE
from ..model_refs import HUB_DIR, HUB_SEPARATOR
from ..project import Project
from .card_writer import CARD_ROLE, unsupported
from .upload import CARD_FILE, publish

EDITED_CARD = "README.edited.md"
TEST_ROWS = 200
EXAMPLES = 5
EDIT_SYSTEM = """You edit a Hugging Face model card as the user asks. Keep every number, table, code block, link and \
the front matter between the --- lines exactly as they are unless the user asks to change them. Never add a number. \
Return the whole card as markdown."""
EDIT_SCHEMA = {"type": "object", "properties": {"markdown": {"type": "string"}}}


def _folder(p: Project, repo: str) -> Path:
    return p.root / HUB_DIR / repo.replace("/", HUB_SEPARATOR)


def preview(name: str, repo: str, license: str = PUBLISH_LICENSE) -> dict[str, Any]:
    from .data_release import releasable
    from .facts import load_facts

    stage = publish(name, repo, license, dry_run=True, log=lambda _: None)
    files = [{"name": f.name, "bytes": f.stat().st_size} for f in sorted(stage.iterdir()) if f.is_file()]
    card = (stage / CARD_FILE).read_text()
    return {"card": card, "files": files, "data": releasable(load_facts(name, repo, license))}


def edit(card: str, instruction: str) -> dict[str, Any]:
    from ..llm import get_llm

    new = (
        get_llm(CARD_ROLE)
        .json(EDIT_SYSTEM, f"Request: {instruction}\n\nCard:\n{card}", EDIT_SCHEMA)
        .get("markdown", "")
    )
    wrong = unsupported(new, card)
    if not new.strip() or wrong:
        return {
            "card": card,
            "refused": f"the edit added numbers that are not in the card: {wrong[:10]}" if wrong else "empty edit",
        }
    return {"card": new, "refused": None}


def save_card(name: str, repo: str, card: str) -> Path:
    out = _folder(Project(name), repo) / EDITED_CARD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card)
    return out


def test_bundle(name: str, text: str | None = None) -> dict[str, Any]:
    from ..data.store import Dataset
    from ..export.bundle import load_runtime

    p = Project(name)
    out = p.root / "export"
    m = load_runtime(out).load(out)
    rows = Dataset(p.data_dir_for(p.spec())).load("test")[:TEST_ROWS]
    t0 = time.perf_counter()
    answers = [m.predict(r["input"])["label"] for r in rows]
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(rows))
    right = sum(a == r["label"] for a, r in zip(answers, rows, strict=True))
    shown = [
        {"input": r["input"], "label": r["label"], "answer": a}
        for r, a in zip(rows[:EXAMPLES], answers[:EXAMPLES], strict=True)
    ]
    mine = m.predict(text) if text else None
    return {"n": len(rows), "right": right, "ms": round(ms, 2), "examples": shown, "answer": mine}
