from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(".byosm") / "app.sqlite"
JOB_FIELDS = ("id", "args", "status", "created", "started", "ended", "code")
SCHEMA = """
create table if not exists jobs (
    id text primary key, args text, status text, created real, started real, ended real, code integer
);
create table if not exists lines (job text, line text);
create index if not exists lines_job on lines (job);
create table if not exists events (job text, data text);
create index if not exists events_job on events (job);
"""


class JobStore:
    def __init__(self, ws: Path) -> None:
        path = ws / DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.executescript(SCHEMA)

    def _write(self, sql: str, params: tuple) -> None:
        with self.lock, self.db:
            self.db.execute(sql, params)

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    def save(self, job: dict[str, Any]) -> None:
        row = {**job, "args": json.dumps(job["args"])}
        marks = ", ".join("?" for _ in JOB_FIELDS)
        self._write(f"insert or replace into jobs values ({marks})", tuple(row[k] for k in JOB_FIELDS))

    def line(self, job_id: str, line: str) -> None:
        self._write("insert into lines values (?, ?)", (job_id, line))

    def event(self, job_id: str, data: dict[str, Any]) -> None:
        self._write("insert into events values (?, ?)", (job_id, json.dumps(data)))

    def jobs(self) -> list[dict[str, Any]]:
        rows = self._read(f"select {', '.join(JOB_FIELDS)} from jobs order by created")
        return [{**dict(zip(JOB_FIELDS, r, strict=True)), "args": json.loads(r[1])} for r in rows]

    def lines(self, job_id: str) -> list[str]:
        return [r[0] for r in self._read("select line from lines where job = ? order by rowid", (job_id,))]

    def events(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._read("select data from events where job = ? order by rowid", (job_id,))
        return [json.loads(r[0]) for r in rows]
