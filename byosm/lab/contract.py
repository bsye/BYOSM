from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Hooks = dict[str, Callable[..., Any]]
CLASSIFIER, FORECASTER = "classifier", "forecaster"
ALLOWED_IMPORTS = frozenset(
    {"collections", "dataclasses", "functools", "itertools", "math", "numpy", "random", "statistics", "torch", "typing"}
)
BLOCKED_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)
BLOCKED_ATTRS = frozenset({"cpp_extension", "hub", "load", "popen", "save", "system"})
SAFE_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "ValueError",
)

HEADER = """A training method is one Python module that defines one or more of these functions, and nothing else
runs at import time (only imports, constants and function definitions):"""
RULES = f"""Allowed imports: {", ".join(sorted(ALLOWED_IMPORTS))}.
No file, network or process access, no eval or exec, no dunder attributes, no torch.hub, torch.load or torch.save."""
AFTER_STEP = """def after_step(model: torch.nn.Module, step: int, total: int) -> None
    Called after every optimiser step, for example to average or perturb weights. step counts optimiser steps
    from 0 and total is the planned steps."""


@dataclass(frozen=True)
class Contract:
    kind: str
    model: str
    goal: str
    won: str
    hooks: tuple[str, ...]
    functions: str
    kept: str

    @property
    def text(self) -> str:
        return f"{HEADER}\n\n{self.functions}\n\n{AFTER_STEP}\n\n{RULES}\n{self.kept}"


CLASSIFIER_CONTRACT = Contract(
    kind=CLASSIFIER,
    model="a small fine-tuned specialist (a classifier or a tagger over text, audio, images or documents, as the task says)",
    goal="the weakness the diagnosis and the failing health checks show",
    won="sure lead",
    hooks=("rows", "loss", "after_step"),
    functions="""def rows(train: list[dict], rng: random.Random) -> list[dict]
    Called once before training with the labelled training rows ({"input": ..., "label": ..., "source": ...,
    "meta": {...}}). "input" is the text for text specialists and a file path for audio, image and document ones;
    "label" is a label name, or one label index per token or frame for taggers. Return the rows to train on:
    reorder them (a curriculum), filter or duplicate them, or rewrite text inputs. Never invent labels or files.

def loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None, step: int) -> torch.Tensor
    Replaces the task loss. logits are (rows, labels): one row per input for classifiers, one per token or frame
    for taggers, where targets of -100 mark positions to ignore. targets are label indices, weight is the per-label
    class weight the trainer computed (or None), step counts optimiser steps from 0. Return a scalar.""",
    kept="The model architecture, tokenizer, evaluation and export stay as they are.",
)
FORECASTER_CONTRACT = Contract(
    kind=FORECASTER,
    model="the patch transformers of a small forecaster (one model trained on every dataset of its domain)",
    goal="a lower calib MASE than the promoted version, to beat the best rival and clear the health checks that are not ok",
    won="promoted",
    hooks=("window", "loss", "after_step"),
    functions="""def window(values: numpy.ndarray, context: int, rng: numpy.random.Generator) -> numpy.ndarray
    Called on every training window before it is scaled. values[:context] is the past the model reads and
    values[context:] the future it learns to forecast; NaN marks a missing value, the padding before a short past
    and the steps after the dataset's horizon. Return an array of the same length: augment, warp or reshape the
    window, but never copy future values into the past.

def loss(pred: torch.Tensor, target: torch.Tensor, known: torch.Tensor, levels: torch.Tensor, step: int) -> torch.Tensor
    Replaces the pinball loss. pred is (batch, quantiles, horizon) in the space of target: the future minus the
    mean of the window's past, divided by the past's standard deviation, clipped to +-50 and, in the v2 recipe,
    passed through arcsinh. known is the 0/1 mask of the target values that exist; both are (batch, horizon).
    levels holds the quantile level of every row of pred, ascending. The default sums max(levels * d, (levels - 1) * d),
    with d = target - pred, over the quantiles and the known values and divides by the number of known values. step
    counts optimiser steps from 0. Return a scalar.""",
    kept="""The patch transformer's architecture, the scaling, the other models, the scoring on calib and test windows
and the saved forecaster stay as they are.""",
)
CONTRACTS = {c.kind: c for c in (CLASSIFIER_CONTRACT, FORECASTER_CONTRACT)}
