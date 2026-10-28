import random

import pytest
import torch

from byosm.data.store import record, write
from byosm.modalities import get_modality
from byosm.spec import Label, TaskSpec, TrainCfg
from byosm.train import build_model, fit
from byosm.train.regularise import mixed_logits, reinit_top, symmetric_kl

LABELS = [Label(name="statement", definition=""), Label(name="question", definition="")]


def _spec(backbone, **train):
    return TaskSpec(
        name="isq", description="", modality="text", labels=LABELS, backbone=backbone, train=TrainCfg(**train)
    )


def _rows(dd):
    rng = random.Random(0)
    for split, n in (("train", 60), ("dev", 20), ("calib", 20), ("test", 20)):
        rows = []
        for i in range(n):
            question = rng.random() < 0.5
            text = f"what about the {i} coffee?" if question else f"i like the {i} coffee."
            rows.append(record(text, "question" if question else "statement", "test"))
        write(dd / f"{split}.jsonl", rows)


def test_the_consistency_loss_is_zero_for_identical_passes_and_positive_otherwise():
    a = torch.tensor([[2.0, -1.0], [0.5, 0.5]])
    assert symmetric_kl(a, a).item() == pytest.approx(0.0, abs=1e-6)
    assert symmetric_kl(a, a.flip(-1)).item() > 0


def test_re_initialising_the_top_layer_changes_it_and_leaves_the_rest(tiny_bert):
    spec = _spec(tiny_bert)
    model = build_model(spec, get_modality(spec, "."))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert reinit_top(model, 1, lambda _: None) == 1
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(".layer.3." in k or ".layers.3." in k for k in changed)


def test_mixed_logits_blend_two_rows_of_the_batch(tiny_bert):
    spec = _spec(tiny_bert)
    mod = get_modality(spec, ".")
    model = build_model(spec, mod).eval()
    inputs, _ = mod.collate(["what about it?", "i like it.", "is it on?"])
    logits, perm, lam = mixed_logits(model, inputs, 0.4, random.Random(0))
    assert logits.shape == (3, 2) and sorted(perm) == [0, 1, 2] and 0 <= lam <= 1


@pytest.mark.parametrize("train", [{"rdrop": 1.0}, {"mixup": 0.4}, {"reinit_layers": 1}])
def test_each_regulariser_trains_a_usable_model(ws, tiny_bert, train):
    _rows(ws / "data")
    spec = _spec(tiny_bert, epochs=6, pretrain_steps=0, batch_size=16, **train)
    meta, _ = fit(spec, ws / "data", ws / "run", log=lambda _: None)
    assert meta["dev"]["objective"] > 0.6
