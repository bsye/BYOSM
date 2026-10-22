from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, insert, select

from ..config import find_workspace
from .engine import session
from .models import Row, Specialist

SUFFIX = ".jsonl"
SPECIALISTS_DIR = "specialists"
INSERT_BATCH = 5000


def _key(path: Path) -> tuple[Path, str | None, str, str]:
    path = path.absolute()
    ws = find_workspace(path.parent)
    parts = path.relative_to(ws).parts if path.is_relative_to(ws) else path.parts
    owner = parts[1] if len(parts) > 2 and parts[0] == SPECIALISTS_DIR else None
    collection = "/".join(parts[2:-1] if owner else parts[:-1])
    return ws, owner, collection, path.name.removesuffix(SUFFIX)


def _owner_id(s, owner: str | None) -> int | None:
    return s.scalar(select(Specialist.id).where(Specialist.name == owner)) if owner else None


def _where(s, owner: str | None, collection: str, split: str | None = None) -> list:
    sid = _owner_id(s, owner)
    clauses = [Row.specialist_id.is_(None) if sid is None else Row.specialist_id == sid, Row.collection == collection]
    return clauses + ([Row.split == split] if split is not None else [])


def read(path: Path) -> list[dict]:
    ws, owner, collection, split = _key(path)
    with session(ws) as s:
        return list(s.scalars(select(Row.data).where(*_where(s, owner, collection, split)).order_by(Row.position)))


def tally(path: Path, key: str) -> list[tuple[Any, int]]:
    ws, owner, collection, split = _key(path)
    value = Row.data[key]
    with session(ws) as s:
        q = select(value, func.count()).where(*_where(s, owner, collection, split)).group_by(value)
        return [(v, n) for v, n in s.execute(q).all()]


def exists(path: Path) -> bool:
    ws, owner, collection, split = _key(path)
    with session(ws) as s:
        return bool(s.scalar(select(func.count()).select_from(Row).where(*_where(s, owner, collection, split))))


def splits(directory: Path, newest_first: bool = False) -> list[str]:
    ws, owner, collection, _ = _key(directory / f"_{SUFFIX}")
    with session(ws) as s:
        q = select(Row.split, func.max(Row.id)).where(*_where(s, owner, collection)).group_by(Row.split)
        found = s.execute(q).all()
    return [n for n, _ in sorted(found, key=lambda t: -t[1])] if newest_first else sorted(n for n, _ in found)


def _insert(s, owner: str | None, collection: str, split: str, rows: list[dict], start: int) -> None:
    sid = _owner_id(s, owner)
    values = [
        {"specialist_id": sid, "collection": collection, "split": split, "position": start + i, "data": r}
        for i, r in enumerate(json.loads(json.dumps(rows, ensure_ascii=False)))
    ]
    for i in range(0, len(values), INSERT_BATCH):
        s.execute(insert(Row), values[i : i + INSERT_BATCH])


def write(path: Path, rows: Iterable[dict]) -> None:
    ws, owner, collection, split = _key(path)
    with session(ws) as s:
        s.execute(delete(Row).where(*_where(s, owner, collection, split)))
        _insert(s, owner, collection, split, list(rows), 0)


def append(path: Path, rows: Iterable[dict]) -> None:
    ws, owner, collection, split = _key(path)
    with session(ws) as s:
        last = s.scalar(select(func.max(Row.position)).where(*_where(s, owner, collection, split)))
        _insert(s, owner, collection, split, list(rows), 0 if last is None else last + 1)


def text(path: Path, limit: int) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in read(path))[:limit]


def remove(directory: Path) -> None:
    ws, owner, collection, _ = _key(directory / f"_{SUFFIX}")
    with session(ws) as s:
        s.execute(delete(Row).where(*_where(s, owner, collection)))
