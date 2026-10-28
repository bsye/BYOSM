from __future__ import annotations

import hashlib
import time
from typing import Any

from ..project import Project
from .static import problems

LAB = "lab:"
PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"
STATUSES = (PENDING, APPROVED, REJECTED)


def draft_id(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


def save_draft(p: Project, name: str, hypothesis: str, code: str, author: str, parent: str | None = None) -> dict:
    did = draft_id(code)
    found = {"id": did, "name": name, "hypothesis": hypothesis, "code": code, "author": author, "parent": parent}
    found |= {"status": PENDING, "problems": problems(code), "created": time.time(), "experiment": None, "error": None}
    p.save_report(LAB + did, found)
    return found


def drafts(p: Project) -> list[dict[str, Any]]:
    return sorted(p.reports(LAB).values(), key=lambda d: -d["created"])


def draft(p: Project, did: str) -> dict[str, Any] | None:
    return p.report(LAB + did)


def update(p: Project, did: str, **fields: Any) -> dict[str, Any]:
    found = draft(p, did)
    if found is None:
        raise KeyError(f"no lab method {did}")
    found |= fields
    p.save_report(LAB + did, found)
    return found


def decide(p: Project, did: str, status: str) -> dict[str, Any]:
    found = draft(p, did)
    if found is None:
        raise KeyError(f"no lab method {did}")
    if status == APPROVED and found["problems"]:
        raise ValueError(f"{did} fails its checks: {'; '.join(found['problems'])}")
    return update(p, did, status=status, decided=time.time())
