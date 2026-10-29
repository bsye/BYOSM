from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..events import EVENTS_ENV, JOB_ENV, TARGET
from .store import JOB_FIELDS

PYTHONPATH = "PYTHONPATH"
TERMINAL_COLUMNS = "160"
ISOLATION = "isolation"


@dataclass
class Job:
    id: str
    args: list[str]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    code: int | None = None
    pid: int | None = None
    cancel: bool = False
    llm: dict[str, Any] | None = None
    target: str | None = None
    stalled: bool = False
    isolation: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (*JOB_FIELDS, TARGET, ISOLATION)}


def job_vars(job: Job) -> dict[str, str]:
    from ..config import LLM_OVERRIDE_ENV

    return {
        "PYTHONUNBUFFERED": "1",
        "COLUMNS": TERMINAL_COLUMNS,
        "NO_COLOR": "1",
        EVENTS_ENV: "1",
        JOB_ENV: job.id,
        **({LLM_OVERRIDE_ENV: json.dumps(job.llm)} if job.llm else {}),
    }


def job_env(ws: Path, job: Job) -> dict[str, str]:
    from ..config import env_file
    from .snapshot import frozen

    code = [str(frozen()), *filter(None, [os.environ.get(PYTHONPATH)])]
    return {**os.environ, **env_file(ws), PYTHONPATH: os.pathsep.join(code), **job_vars(job)}
