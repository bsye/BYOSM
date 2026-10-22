from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import TaskSpec
from .base import Predictor, register_baseline

FINISHED_NAMES = ("finished", "complete", "completed", "end_of_turn", "done", "eou", "yes")
PLUGIN = "livekit.plugins.turn_detector"
RUNNER_MODULES = (f"{PLUGIN}.english", f"{PLUGIN}.multilingual")
HUB_REPO = "livekit/turn-detector"
HUB_URL = f"https://huggingface.co/{HUB_REPO}"
LANGUAGES_FILE = "languages.json"
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

    if importlib.util.find_spec(PLUGIN):
        for module in RUNNER_MODULES:
            importlib.import_module(module)


class LiveKitTurn(Predictor):
    modalities, task_types = ("text",), ("classify",)
    hub_repo = HUB_REPO
    variant = "english"
    runner_class = "_EUORunnerEn"
    own_threshold = False

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        import importlib

        try:
            runner = getattr(importlib.import_module(f"{PLUGIN}.{self.variant}"), self.runner_class)
        except ImportError as e:
            raise ImportError("pip install livekit-plugins-turn-detector transformers") from e
        self.finished = spec.label_index[finished_label(spec)]
        runner._download_files()
        self.runner = runner()
        self.runner.initialize()
        self.name = f"livekit/turn-detector {self.variant} {runner.model_revision()}"
        self.threshold = _language_threshold(runner, spec.languages[0]) if self.own_threshold else None

    def _eou(self, text: Any) -> float:
        out = self.runner.run(json.dumps({"chat_ctx": chat_turns(text)}).encode())
        return float(json.loads(out or b"{}").get("eou_probability", 0.0))

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        p = np.array([self._eou(x) for x in raws], np.float32)
        if self.threshold is not None:
            p = (p >= self.threshold).astype(np.float32)
        out = np.zeros((len(raws), 2), np.float32)
        out[:, self.finished] = p
        out[:, 1 - self.finished] = 1 - p
        return out


def _language_threshold(runner: Any, language: str) -> float:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        runner.hub_repo if hasattr(runner, "hub_repo") else HUB_REPO, LANGUAGES_FILE, revision=runner.model_revision()
    )
    found = json.loads(Path(path).read_text()).get(language)
    if not found:
        raise ValueError(f"LiveKit's turn detector has no tuned threshold for {language!r}")
    return float(found["threshold"])


register_baseline(
    "livekit-turn",
    "LiveKit's English-only end-of-turn model (deprecated by LiveKit), the exact ONNX and preprocessing its plugin "
    "ships, with its threshold calibrated on our calib split like every rival.",
)(LiveKitTurn)


@register_baseline(
    "livekit-turn-default",
    "LiveKit's English-only end-of-turn model at LiveKit's own tuned threshold for the language, not ours.",
)
class LiveKitTurnDefault(LiveKitTurn):
    own_threshold = True
    hub_repo, home = None, HUB_URL


@register_baseline(
    "livekit-turn-multilingual",
    "LiveKit's multilingual end-of-turn model, the one LiveKit recommends, with its threshold calibrated on our calib "
    "split like every rival.",
)
class LiveKitTurnMultilingual(LiveKitTurn):
    variant = "multilingual"
    runner_class = "_EUORunnerMultilingual"


@register_baseline(
    "livekit-turn-multilingual-default",
    "LiveKit's multilingual end-of-turn model at LiveKit's own tuned threshold for the language, not ours.",
)
class LiveKitTurnMultilingualDefault(LiveKitTurnMultilingual):
    own_threshold = True
    hub_repo, home = None, HUB_URL
