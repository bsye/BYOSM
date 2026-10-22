from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..spec import TaskSpec
from .base import Predictor, register_baseline

FINISHED_NAMES = ("finished", "complete", "completed", "end_of_turn", "done", "eou", "yes")
RUNNER_MODULE = "livekit.plugins.turn_detector.english"
SPEAKERS = {"user": "user", "agent": "assistant", "assistant": "assistant", "bot": "assistant"}


def finished_label(spec: TaskSpec) -> str:
    if len(spec.labels) != 2:
        raise ValueError("the LiveKit turn detector answers one question, finished or not: it needs exactly two labels")
    name = spec.positive_label or next((n for n in spec.label_names if n.lower() in FINISHED_NAMES), None)
    if not name:
        raise ValueError(f"set positive_label to the label that means the turn is over (one of {spec.label_names})")
    return name


def chat_turns(text: str) -> list[dict[str, str]]:
    turns = []
    for line in str(text).splitlines():
        who, sep, said = line.partition(":")
        role = SPEAKERS.get(who.strip().lower()) if sep else None
        if role:
            turns.append({"role": role, "content": said.strip()})
        elif turns:
            turns[-1]["content"] += " " + line.strip()
        else:
            turns.append({"role": "user", "content": line.strip()})
    return turns or [{"role": "user", "content": str(text)}]


def register_on_main_thread() -> None:
    import importlib.util

    if importlib.util.find_spec("livekit.plugins.turn_detector"):
        importlib.import_module(RUNNER_MODULE)


@register_baseline(
    "livekit-turn",
    "LiveKit's end-of-turn model, the exact ONNX and preprocessing livekit-plugins-turn-detector ships (English).",
)
class LiveKitTurn(Predictor):
    modalities, task_types = ("text",), ("classify",)
    hub_repo = "livekit/turn-detector"

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        try:
            from livekit.plugins.turn_detector.english import _EUORunnerEn
        except ImportError as e:
            raise ImportError("pip install livekit-plugins-turn-detector transformers") from e
        self.finished = spec.label_index[finished_label(spec)]
        _EUORunnerEn._download_files()
        self.runner = _EUORunnerEn()
        self.runner.initialize()
        self.name = "livekit/turn-detector " + _EUORunnerEn.model_revision()

    def _eou(self, text: Any) -> float:
        out = self.runner.run(json.dumps({"chat_ctx": chat_turns(text)}).encode())
        return float(json.loads(out or b"{}").get("eou_probability", 0.0))

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        p = np.array([self._eou(x) for x in raws], np.float32)
        out = np.zeros((len(raws), 2), np.float32)
        out[:, self.finished] = p
        out[:, 1 - self.finished] = 1 - p
        return out
