from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .distill import ABSTRACTION, MODEL
from .infosets import PAD, layout
from .teacher import command, ensure_image, run, strategy, work_dir

CPUS = "2"
MEMORY = "1g"
STAY_OPEN = ("-i",)
Log = Callable[[str], None]


def abstraction(ws: Path, hands_dir: Path | None, name: str, log: Log) -> str:
    saved = f"{name}{ABSTRACTION}"
    if not (work_dir(ws) / saved).exists():
        run(ws, hands_dir, ["abstraction", "--blueprint", strategy(ws, name), "--out", saved], log)
    return saved


def student(ws: Path, name: str) -> Any:
    import torch

    from .student import STREETS, Net

    path = work_dir(ws) / f"{name}{MODEL}"
    if not path.exists():
        raise ValueError(f"no student model named {name!r}: byosm poker student trains one and prints its name")
    state = torch.load(path, map_location="cpu")
    net = Net(state["cards.weight"].shape[0] // STREETS)
    net.load_state_dict(state)
    return net.eval()


@dataclass
class Advisor:
    net: Any
    engine: Any

    @classmethod
    def start(cls, ws: Path, hands_dir: Path | None, name: str, log: Log) -> Advisor:
        ensure_image(log)
        args = ["spots", "--abs", abstraction(ws, hands_dir, name, log)]
        piped = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "text": True}
        return cls(student(ws, name), subprocess.Popen(command(ws, hands_dir, args, CPUS, MEMORY, STAY_OPEN), **piped))

    def spot(self, hand: str) -> dict:
        self.engine.stdin.write(json.dumps(hand) + "\n")
        self.engine.stdin.flush()
        answered = self.engine.stdout.readline()
        if not answered:
            raise OSError("the engine closed its output")
        found = json.loads(answered)
        if "error" in found:
            raise ValueError(found["error"])
        return found

    def advise(self, hand: str) -> dict:
        import torch

        from .student import inputs

        found = self.spot(hand)
        row = np.zeros(1, layout(len(found["offered"]), max(len(found["hist"]), 1)))
        row["bucket"], row["actions"], row["tokens"] = found["bucket"], len(found["offered"]), len(found["hist"])
        row["offered"][0], row["hist"][0] = found["offered"], PAD
        row["hist"][0, : len(found["hist"])] = found["hist"]
        with torch.no_grad():
            probs = torch.softmax(self.net(**inputs(row, self.net.span, "cpu")), dim=-1)[0]
        moves = [
            {"move": m, "probability": float(probs[slot])}
            for m, slot in zip(found["moves"], found["offered"], strict=True)
        ]
        return {"seat": found["seat"], "street": found["street"].lower(), "moves": moves}

    def close(self) -> None:
        self.engine.stdin.close()
        self.engine.wait()
