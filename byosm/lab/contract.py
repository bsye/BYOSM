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
    Called once before training with the labelled training rows ({"input": ..., "label": ..., "source": ...,
    "meta": {...}}). "input" is the text for text specialists and a file path for audio, image and document ones;
    "label" is a label name, or one label index per token or frame for taggers. Return the rows to train on:
    reorder them (a curriculum), filter or duplicate them, or rewrite text inputs. Never invent labels or files.

def loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None, step: int) -> torch.Tensor
    Replaces the task loss. logits are (rows, labels): one row per input for classifiers, one per token or frame
    for taggers, where targets of -100 mark positions to ignore. targets are label indices, weight is the per-label
    class weight the trainer computed (or None), step counts optimiser steps from 0. Return a scalar.

def after_step(model: torch.nn.Module, step: int, total: int) -> None
    Called after every optimiser step, for example to average or perturb weights. total is the planned steps.

Allowed imports: collections, dataclasses, functools, itertools, math, numpy, random, statistics, torch, typing.
No file, network or process access, no eval or exec, no dunder attributes, no torch.hub, torch.load or torch.save.
The model architecture, tokenizer, evaluation and export stay as they are."""
