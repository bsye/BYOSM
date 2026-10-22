from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Project
from ..events import emit

DATA_BUDGET_SHARE = 0.4


class Log:
    def __init__(self, p: Project, echo: Callable[[str], None]):
        self.p, self.echo, self.decisions = p, echo, []
        f = p.root / "auto.json"
        if f.exists():
            self.decisions = json.loads(f.read_text()).get("decisions", [])

    def __call__(self, step: str, choice: Any, why: str, **extra: Any) -> None:
        d = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "choice": choice, "why": why, **extra}
        self.decisions.append(d)
        emit("decision", **{k: v for k, v in d.items() if k != "t"})
        self.echo(f"[auto] {step}: {choice} - {why}")
        self.save()

    def save(self, final: dict | None = None) -> None:
        self.p.root.mkdir(parents=True, exist_ok=True)
        (self.p.root / "auto.json").write_text(
            json.dumps({"decisions": self.decisions, "final": final}, indent=2, default=str)
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
