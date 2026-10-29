from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, func, select, type_coerce
from sqlalchemy.orm import Session, sessionmaker

from ..config import STATE_DIR

URL_ENV = "DATABASE_URL"
LOCAL_DB = STATE_DIR / "byosm.sqlite"
SQLITE_TIMEOUT_S = 30


def database_url(ws: Path) -> str:
    return os.environ.get(URL_ENV) or f"sqlite:///{ws / LOCAL_DB}"


def _finite(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_finite(x) for x in v]
    return v


def _json(v: Any) -> str:
    return json.dumps(_finite(v), ensure_ascii=False)


def _sqlite_pragmas(conn, _) -> None:
    cur = conn.cursor()
    cur.execute("pragma journal_mode=wal")
    cur.execute("pragma foreign_keys=on")
    cur.close()


@cache
def engine(url: str) -> Engine:
    from .models import Base

    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        args = {"timeout": SQLITE_TIMEOUT_S, "check_same_thread": False}
        eng = create_engine(url, connect_args=args, json_serializer=_json)
        event.listen(eng, "connect", _sqlite_pragmas)
    else:
        eng = create_engine(url, pool_pre_ping=True, json_serializer=_json)
    Base.metadata.create_all(eng)
    return eng


@cache
def _maker(url: str) -> sessionmaker[Session]:
    return sessionmaker(engine(url), expire_on_commit=False)


@contextmanager
def session(ws: Path) -> Iterator[Session]:
    with _maker(database_url(ws)).begin() as s:
        yield s


def json_without(s: Session, column: Any, key: str) -> Any:
    from .models import Json

    dropped = column.op("-")(key) if s.get_bind().dialect.name == "postgresql" else func.json_remove(column, f"$.{key}")
    return type_coerce(dropped, Json)


def get_or_add(s: Session, model: type, **keys: object) -> object:
    row = s.scalar(select(model).filter_by(**keys))
    if row is None:
        row = model(**keys)
        s.add(row)
    return row
