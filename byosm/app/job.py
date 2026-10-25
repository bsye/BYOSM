from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..events import EVENTS_ENV, TARGET
from ..train.gpu import JOB_ENV
from .store import JOB_FIELDS

PYTHONPATH = "PYTHONPATH"
TERMINAL_COLUMNS = "160"


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

    def public(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (*JOB_FIELDS, TARGET)}


def job_env(ws: Path, job: Job) -> dict[str, str]:
    from ..config import LLM_OVERRIDE_ENV, env_file
    from .snapshot import frozen

    code = [str(frozen()), *filter(None, [os.environ.get(PYTHONPATH)])]
    return {
        **os.environ,
        **env_file(ws),
        PYTHONPATH: os.pathsep.join(code),
        "PYTHONUNBUFFERED": "1",
        "COLUMNS": TERMINAL_COLUMNS,
        "NO_COLOR": "1",
        EVENTS_ENV: "1",
        JOB_ENV: job.id,
        **({LLM_OVERRIDE_ENV: json.dumps(job.llm)} if job.llm else {}),
    }
