from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(".byosm") / "app.sqlite"
JOB_FIELDS = ("id", "args", "status", "created", "started", "ended", "code", "pid")
SCHEMA = """
create table if not exists jobs (
    id text primary key, args text, status text, created real, started real, ended real, code integer, pid integer
);
create table if not exists lines (job text, line text);
create index if not exists lines_job on lines (job);
create table if not exists events (job text, data text);
create index if not exists events_job on events (job);
"""


USAGE_COLUMNS = (
    "job",
    "role",
    "model",
    "calls",
    "failed",
    "tokens_in",
    "tokens_out",
    "seconds",
    "last",
    "counted",
    "counted_seconds",
)
USAGE_SQL = """
select job, json_extract(data, '$.role'), json_extract(data, '$.model'),
  sum(json_extract(data, '$.state') = 'done'), sum(json_extract(data, '$.state') = 'failed'),
  coalesce(sum(json_extract(data, '$.tokens_in')), 0), coalesce(sum(json_extract(data, '$.tokens_out')), 0),
  round(coalesce(sum(json_extract(data, '$.seconds')), 0), 1), max(json_extract(data, '$.at')),
  sum(json_extract(data, '$.tokens_out') is not null),
  round(coalesce(sum(case when json_extract(data, '$.tokens_out') is not null then json_extract(data, '$.seconds') end), 0), 1)
from events
where json_extract(data, '$.kind') = 'llm' and json_extract(data, '$.state') in ('done', 'failed')
group by 1, 2, 3
order by 9 desc
"""


class JobStore:
    def __init__(self, ws: Path) -> None:
        path = ws / DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.executescript(SCHEMA)
            columns = [r[1] for r in self.db.execute("pragma table_info(jobs)")]
            if "pid" not in columns:
                self.db.execute("alter table jobs add column pid integer")

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

    def seen(self, job_id: str) -> int:
        counts = [self._read(f"select count(*) from {t} where job = ?", (job_id,))[0][0] for t in ("lines", "events")]
        return sum(counts)

    def events(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._read("select data from events where job = ? order by rowid", (job_id,))
        return [json.loads(r[0]) for r in rows]

    def llm_usage(self) -> list[dict[str, Any]]:
        rows = self._read(USAGE_SQL)
        return [dict(zip(USAGE_COLUMNS, r, strict=True)) for r in rows]

    def jobs_using(self, model: str) -> set[str]:
        sql = "select distinct job from events where json_extract(data, '$.kind') = 'llm' and json_extract(data, '$.model') = ?"
        return {r[0] for r in self._read(sql, (model,))}
