import byosm.modalities.base as base
from byosm.improve.paths import untrainable


def test_only_hub_backbones_without_torch_weights_are_untrainable(monkeypatch):
    monkeypatch.setattr(base, "trainable", lambda repo: repo != "org/onnx-only")
    ops = [
        {"op": "set", "path": "backbone", "value": "hf:org/onnx-only"},
        {"op": "set", "path": "document.text_backbone", "value": "hf:org/torch"},
        {"op": "set", "path": "train.epochs", "value": 4},
    ]
    assert untrainable(ops) == ["hf:org/onnx-only"]
