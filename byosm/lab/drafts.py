from __future__ import annotations

import hashlib
import time
from typing import Any

from .owners import Owner, kind_of
from .static import problems

LAB = "lab:"
PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"
STATUSES = (PENDING, APPROVED, REJECTED)


class RefusedError(ValueError):
    pass


def draft_id(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


def save_draft(owner: Owner, name: str, hypothesis: str, code: str, author: str, parent: str | None = None) -> dict:
    did, kind = draft_id(code), kind_of(owner)
    found = {"id": did, "kind": kind, "name": name, "hypothesis": hypothesis, "code": code, "author": author}
    found |= {"parent": parent, "status": PENDING, "problems": problems(code, kind), "created": time.time()}
    found |= {"experiment": None, "error": None}
    owner.save_report(LAB + did, found)
    return found


def drafts(owner: Owner) -> list[dict[str, Any]]:
    return sorted(owner.reports(LAB).values(), key=lambda d: -d["created"])


def draft(owner: Owner, did: str) -> dict[str, Any] | None:
    return owner.report(LAB + did)


def update(owner: Owner, did: str, **fields: Any) -> dict[str, Any]:
    found = draft(owner, did)
    if found is None:
        raise KeyError(f"no lab method {did}")
    found |= fields
    owner.save_report(LAB + did, found)
    return found


def decide(owner: Owner, did: str, status: str) -> dict[str, Any]:
    found = draft(owner, did)
    if found is None:
        raise KeyError(f"no lab method {did}")
    if status == APPROVED and found["problems"]:
        raise ValueError(f"{did} fails its checks: {'; '.join(found['problems'])}")
    return update(owner, did, status=status, decided=time.time())
