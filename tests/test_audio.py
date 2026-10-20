import numpy as np
import torch

from byosm.data.audio_mix import Mixer, frame_activity
from byosm.modalities.audio import TinyCRNN
from byosm.spec import Label, TaskSpec

CHUNK = 480


def test_crnn_frames_align_with_hop():
    m = TinyCRNN(16000, 10.0)
    wav = torch.randn(2, 16000)
    feats, mask = m(wav, torch.tensor([16000, 8000]))
    assert feats.shape[1] == 100 and mask[0].sum() == 100 and mask[1].sum() == 50


def test_crnn_streaming_matches_offline():
    torch.manual_seed(0)
    m = TinyCRNN(16000, 10.0).eval()
    wav = torch.randn(1, 16000) * 0.1
    full, _ = m(wav, torch.tensor([16000]))
    state, outs = None, []
    for i in range(0, 16000, CHUNK):
        f, state = m.stream(wav[:, i : i + CHUNK], state)
        outs.append(f)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4)


def test_mixer_labels_follow_events():
    spec = TaskSpec(
        name="vad",
        description="",
        modality="audio",
        task_type="frame",
        labels=[Label(name="silence", definition=""), Label(name="speech", definition="")],
        positive_label="speech",
        backbone="tiny-crnn",
    )
    t = np.arange(16000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, 0.01, 32000).astype(np.float32)
    mx = Mixer(spec, [tone], [noise], seed=1)
    for _ in range(20):
        wav, frames = mx.clip()
        assert len(frames) == len(wav) // 160
    act = frame_activity(np.concatenate([np.zeros(1600, np.float32), tone[:3200]]), 160)
    assert act[:10].sum() == 0 and act[10:30].all()
