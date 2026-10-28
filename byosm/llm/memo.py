from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from typing import Any

MEMO_DB = "llm-memo.sqlite"
QUERY_CHUNK = 500
LOCK_WAIT_S = 30.0
SEPARATOR = "\x1f"


def memo_key(*parts: str) -> str:
    return hashlib.sha1(SEPARATOR.join(parts).encode()).hexdigest()


def _db() -> sqlite3.Connection:
    from ..config import cache_root

    cache_root().mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache_root() / MEMO_DB, timeout=LOCK_WAIT_S)
    db.execute("create table if not exists memo (key text primary key, value text)")
    return db


def recall(keys: list[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    with closing(_db()) as db:
        for i in range(0, len(keys), QUERY_CHUNK):
            part = keys[i : i + QUERY_CHUNK]
            query = f"select key, value from memo where key in ({','.join('?' * len(part))})"
            found |= {k: json.loads(v) for k, v in db.execute(query, part)}
    return found


def keep(values: dict[str, Any]) -> None:
    with closing(_db()) as db, db:
        db.executemany("insert or replace into memo values (?, ?)", [(k, json.dumps(v)) for k, v in values.items()])


def memoised(
    keys: list[str],
    compute: Callable[[list[str]], dict[str, Any]],
    batch: int,
    workers: int,
    log: Callable[[str], None],
) -> list[Any]:
    from . import pmap

    unique = list(dict.fromkeys(keys))
    found = recall(unique)
    todo = [k for k in unique if k not in found]

    def run(part: list[str]) -> dict[str, Any]:
        fresh = compute(part)
        keep(fresh)
        return fresh

    parts = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    for fresh in pmap(run, parts, workers, on_error=lambda e: log(f"  LLM batch failed: {e}")):
        found |= fresh
    return [found.get(k) for k in keys]
