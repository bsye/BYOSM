from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

URL_ENV = "DATABASE_URL"
LOCAL_DB = Path(".byosm") / "byosm.sqlite"
SQLITE_TIMEOUT_S = 30


def database_url(ws: Path) -> str:
    return os.environ.get(URL_ENV) or f"sqlite:///{ws / LOCAL_DB}"


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
        eng = create_engine(url, connect_args={"timeout": SQLITE_TIMEOUT_S, "check_same_thread": False})
        event.listen(eng, "connect", _sqlite_pragmas)
    else:
        eng = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(eng)
    return eng


@cache
def _maker(url: str) -> sessionmaker[Session]:
    return sessionmaker(engine(url), expire_on_commit=False)


@contextmanager
def session(ws: Path) -> Iterator[Session]:
    with _maker(database_url(ws)).begin() as s:
        yield s


def get_or_add(s: Session, model: type, **keys: object) -> object:
    row = s.scalar(select(model).filter_by(**keys))
    if row is None:
        row = model(**keys)
        s.add(row)
    return row
