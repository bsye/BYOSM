from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..events import emit
from ..project import Project

DATA_BUDGET_SHARE = 0.4
AUTOPILOT = "autopilot"


class Log:
    def __init__(self, p: Project, echo: Callable[[str], None]):
        self.p, self.echo = p, echo
        self.decisions = (p.report(AUTOPILOT) or {}).get("decisions", [])

    def __call__(self, step: str, choice: Any, why: str, **extra: Any) -> None:
        d = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "choice": choice, "why": why, **extra}
        self.decisions.append(d)
        emit("decision", **{k: v for k, v in d.items() if k != "t"})
        self.echo(f"[auto] {step}: {choice} - {why}")
        self.save()

    def save(self, final: dict | None = None) -> None:
        self.p.save_report(
            AUTOPILOT, json.loads(json.dumps({"decisions": self.decisions, "final": final}, default=str))
        )


@dataclass
class Session:
    name: str
    p: Project
    log: Log
    base: Any
    margin: float
    deadline: float
    echo: Callable[[str], None]
    durations: list[float] = field(default_factory=list)

    def mean_duration(self) -> float:
        return sum(self.durations) / max(1, len(self.durations))

    def data_seconds(self) -> float:
        return DATA_BUDGET_SHARE * max(0.0, self.deadline - time.time())
