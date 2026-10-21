from __future__ import annotations

from ..spec import TaskSpec
from .audio import EnergyVAD, SileroVAD
from .base import BASELINES, Predictor, register_baseline
from .experiment import ExperimentPredictor
from .hf import HFClassifier, HFTagger
from .jev import Jev
from .zero_shot import ZeroShotLLM

__all__ = [
    "BASELINES",
    "EnergyVAD",
    "ExperimentPredictor",
    "HFClassifier",
    "HFTagger",
    "Jev",
    "Predictor",
    "SileroVAD",
    "ZeroShotLLM",
    "get_baseline",
    "register_baseline",
]


def get_baseline(name: str, spec: TaskSpec) -> Predictor:
    if name.startswith("hf:"):
        tagger = spec.task_type == "frame" and spec.modality == "text"
        return (HFTagger if tagger else HFClassifier)(spec, name[3:])
    if name.startswith("llm:"):
        return ZeroShotLLM(spec, role=name[4:])
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    cls = BASELINES[name]
    if cls.modalities and spec.modality not in cls.modalities or spec.task_type not in cls.task_types:
        supported = f"{cls.modalities} / {cls.task_types}"
        raise ValueError(f"baseline {name!r} supports {supported}, not {spec.modality} / {spec.task_type}")
    return cls(spec)
