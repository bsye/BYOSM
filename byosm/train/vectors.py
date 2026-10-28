from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable

import numpy as np

from ..config import cache_db
from ..spec import TaskSpec

CACHE_DB = "embeddings.sqlite"
TABLE = "vectors (encoder text, item text, vec blob, primary key (encoder, item))"
LOOKUP_CHUNK = 500


def encoder_key(spec: TaskSpec) -> str:
    return f"{spec.backbone}|{spec.train.max_len}|{spec.train.truncation}"


def _item(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _known(db: sqlite3.Connection, encoder: str, items: list[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for i in range(0, len(items), LOOKUP_CHUNK):
        part = items[i : i + LOOKUP_CHUNK]
        query = f"select item, vec from vectors where encoder = ? and item in ({', '.join('?' * len(part))})"
        found.update(db.execute(query, (encoder, *part)))
    return found


def cached_embeddings(
    spec: TaskSpec, texts: list[str], encode: Callable[[list[str]], np.ndarray], log: Callable[[str], None]
) -> np.ndarray:
    db, encoder = cache_db(CACHE_DB, TABLE), encoder_key(spec)
    items = [_item(t) for t in texts]
    found = _known(db, encoder, sorted(set(items)))
    missing = {k: t for k, t in zip(items, texts, strict=True) if k not in found}
    log(f"  embeddings: {len(missing)} encoded, {len(set(items)) - len(missing)} cached")
    if missing:
        vecs = encode(list(missing.values())).astype(np.float32)
        fresh = {k: v.tobytes() for k, v in zip(missing, vecs, strict=True)}
        db.executemany("insert or replace into vectors values (?, ?, ?)", [(encoder, k, v) for k, v in fresh.items()])
        db.commit()
        found.update(fresh)
    db.close()
    return np.stack([np.frombuffer(found[k], np.float32) for k in items])
