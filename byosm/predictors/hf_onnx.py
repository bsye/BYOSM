from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ONNX_FILES = ("model.onnx", "onnx/model.onnx", "model_quant.onnx", "onnx/model_quantized.onnx")


def onnx_file(repo: str) -> str | None:
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(repo))
    return next((f for f in ONNX_FILES if f in files), None)


class OnnxClassifier:
    def __init__(self, repo: str, filename: str):
        import onnx
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig

        path = hf_hub_download(repo, filename)
        self.config = AutoConfig.from_pretrained(repo)
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.n_params = sum(int(np.prod(t.dims)) for t in onnx.load(path, load_external_data=False).graph.initializer)

    def eval(self) -> OnnxClassifier:
        return self

    def parameters(self) -> list[torch.Tensor]:
        return [torch.empty(self.n_params)]

    def __call__(self, **enc: Any) -> SimpleNamespace:
        feed = {k: v.numpy().astype(np.int64) for k, v in enc.items() if k in self.inputs}
        return SimpleNamespace(logits=torch.from_numpy(self.session.run(None, feed)[0]))
