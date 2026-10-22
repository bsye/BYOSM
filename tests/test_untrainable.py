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


def test_a_proposal_that_makes_an_invalid_spec_is_dropped():
    from byosm.improve.proposals import _valid
    from byosm.spec import Label, TaskSpec

    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    bad = {"id": "axes", "ops": [{"op": "set", "path": "diversity_axes", "value": ["register: terse"]}]}
    good = {"id": "epochs", "ops": [{"op": "set", "path": "train.epochs", "value": 4}]}
    assert not _valid(spec, bad, lambda _: None)
    assert _valid(spec, good, lambda _: None)
