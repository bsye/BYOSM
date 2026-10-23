from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..events import EVENTS_ENV, TARGET
from .store import JOB_FIELDS

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

    return {
        **os.environ,
        **env_file(ws),
        "PYTHONUNBUFFERED": "1",
        "COLUMNS": TERMINAL_COLUMNS,
        "NO_COLOR": "1",
        EVENTS_ENV: "1",
        **({LLM_OVERRIDE_ENV: json.dumps(job.llm)} if job.llm else {}),
    }
