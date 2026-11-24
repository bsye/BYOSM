from __future__ import annotations

import json
import os
from pathlib import Path

from .infosets import SLOTS
from .teacher import work_dir

GRAPH = ".onnx"
ENCODING = ".seat.json"
FRESH = ".tmp"
OPSET = 17
INPUTS = ("tokens", "last", "cards", "legal")
OUTPUT = "logits"
TOKENS_AXIS = {1: "tokens"}


def encoding(span: int) -> dict[str, int]:
    from .student import SEPARATOR, START

    return {"start": START, "separator": SEPARATOR, "slots": SLOTS, "span": span}


def exported(ws: Path, name: str) -> tuple[Path, Path]:
    import torch

    from .advisor import student
    from .student import START

    graph, meta = (work_dir(ws) / f"{name}{suffix}" for suffix in (GRAPH, ENCODING))
    if graph.exists() and meta.exists():
        return graph, meta
    net = student(ws, name)
    one = torch.zeros(1, dtype=torch.long)
    example = (torch.full((1, 2), START), one, one, torch.ones(1, SLOTS, dtype=torch.bool))
    fresh = graph.with_name(f"{graph.name}{FRESH}")
    torch.onnx.export(
        net,
        example,
        str(fresh),
        input_names=list(INPUTS),
        output_names=[OUTPUT],
        dynamic_axes={"tokens": TOKENS_AXIS},
        opset_version=OPSET,
        dynamo=False,
    )
    os.replace(fresh, graph)
    meta.write_text(json.dumps(encoding(net.span)))
    return graph, meta
