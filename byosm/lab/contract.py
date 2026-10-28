from __future__ import annotations

HOOKS = ("rows", "loss", "after_step")
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

CONTRACT = """A training method is one Python module that defines one or more of these functions, and nothing else
runs at import time (only imports, constants and function definitions):

def rows(train: list[dict], rng: random.Random) -> list[dict]
    Called once before training with the labelled training rows ({"input": text, "label": name, "source": ...,
    "meta": {...}}). Return the rows to train on: reorder them (a curriculum), filter, duplicate or rewrite inputs.
    Never invent labels for rows you did not see; keep "label" values from the label set.

def loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None, step: int) -> torch.Tensor
    Replaces the classification loss. logits are (batch, labels), targets are label indices, weight is the
    per-label class weight the trainer computed (or None), step counts optimiser steps from 0. Return a scalar.

def after_step(model: torch.nn.Module, step: int, total: int) -> None
    Called after every optimiser step, for example to average or perturb weights. total is the planned steps.

Allowed imports: collections, dataclasses, functools, itertools, math, numpy, random, statistics, torch, typing.
No file, network or process access, no eval or exec, no dunder attributes, no torch.hub, torch.load or torch.save.
The model architecture, tokenizer, evaluation and export stay as they are."""
