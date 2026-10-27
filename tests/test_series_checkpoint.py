import numpy as np
import pytest
import torch

import byosm.series.checkpoint as checkpoint
import byosm.series.patch_net as patch_net
from byosm.series.patch_net import SMALL, PatchNet, picker, train

STEPS = 6


class StoppedError(Exception):
    pass


def _series():
    t = np.arange(24 * 40)
    return [(10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(i).normal(0, 0.1, len(t)), 48) for i in range(3)]


def _train(resume=None, report=None):
    torch.manual_seed(0)
    net = PatchNet(SMALL)
    train(net, picker(_series(), SMALL.patch), STEPS, 1e-3, 4, "cpu", 0, report, resume)
    return net


def test_an_interrupted_training_resumes_from_its_checkpoint_and_ends_with_the_same_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "EVERY", 2)
    monkeypatch.setattr(patch_net, "REPORT_EVERY", 1)
    whole = _train()

    def stop_at_five(step, loss):
        if step == 5:
            raise StoppedError

    saved = tmp_path / "run.pt"
    with pytest.raises(StoppedError):
        _train(saved, stop_at_five)
    assert torch.load(saved, weights_only=False)["step"] == 4

    resumed = _train(saved)
    assert not saved.exists()
    for a, b in zip(whole.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.allclose(a, b)


def test_the_checkpoint_key_changes_with_the_training_data():
    a, b = np.arange(5.0), np.arange(6.0)
    assert checkpoint.key("x", 1, arrays=[a]) == checkpoint.key("x", 1, arrays=[a.copy()])
    assert checkpoint.key("x", 1, arrays=[a]) != checkpoint.key("x", 1, arrays=[b])
