import json
import random

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from byosm.data.store import record, write
from byosm.distill import TINY_TEXT, student_spec
from byosm.export import export
from byosm.modalities import get_modality
from byosm.pipeline import build
from byosm.project import WEIGHTS, Project
from byosm.spec import Label, TaskSpec, TrainCfg
from byosm.train import build_model, fit
from byosm.train.embedding import ABSENT_LOGIT, logistic_weights
from byosm.train.vectors import cached_embeddings

WORDS = ["pasta", "train", "python", "rain", "museum", "bug", "coffee", "flight", "merge", "garden"]
LABELS = [Label(name="statement", definition="not a question"), Label(name="question", definition="asks something")]


def utterance(rng):
    w = " ".join(rng.sample(WORDS, rng.randint(2, 5)))
    text, label = (f"what about {w}?", "question") if rng.random() < 0.5 else (f"i like {w}.", "statement")
    return record(text, label, "test")


def head_spec(backbone, name="isq", method="embedding-head", **train):
    return TaskSpec(
        name=name,
        description="detect questions",
        modality="text",
        labels=LABELS,
        backbone=backbone,
        finalist_seeds=1,
        train=TrainCfg(method=method, **train),
    )


def write_rows(dd, sizes=(("train", 80), ("dev", 30), ("calib", 30), ("test", 30))):
    rng = random.Random(0)
    for split, n in sizes:
        write(dd / f"{split}.jsonl", [utterance(rng) for _ in range(n)])


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.mark.parametrize("classes", [[0, 2], [0, 1, 3]])
def test_a_logistic_head_becomes_linear_weights_with_the_same_probabilities(classes):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.array([classes[i % len(classes)] for i in range(60)])
    clf = LogisticRegression(max_iter=1000).fit(x, y)
    w, b = logistic_weights(clf, 4)
    probs = torch.softmax(torch.from_numpy(x @ w.T + b), -1).numpy()
    absent = [c for c in range(4) if c not in classes]
    assert np.allclose(probs[:, classes], clf.predict_proba(x), atol=1e-5)
    assert (b[absent] == ABSENT_LOGIT).all() and probs[:, absent].max() < 1e-6


@pytest.mark.parametrize(
    "change",
    [{"modality": "audio"}, {"task_type": "frame"}, {"train": {"method": "embedding-head", "pool": "attention"}}],
)
def test_an_embedding_head_trains_mean_pooled_text_classifiers_only(change):
    with pytest.raises(ValidationError, match="text classifiers only"):
        TaskSpec.model_validate({**head_spec("hf:x").model_dump(), **change})


def test_each_text_is_encoded_once_per_encoder(cache):
    calls = []

    def encode(texts):
        calls.append(texts)
        return np.array([[len(t), 1.0] for t in texts], np.float32)

    spec = head_spec("hf:x")
    first = cached_embeddings(spec, ["ab", "abc", "ab"], encode, lambda _: None)
    again = cached_embeddings(spec, ["abc", "abcd"], encode, lambda _: None)
    cached_embeddings(head_spec("hf:x", max_len=64), ["ab"], encode, lambda _: None)
    assert calls == [["ab", "abc"], ["abcd"], ["ab"]]
    assert first[:, 0].tolist() == [2, 3, 2] and again[:, 0].tolist() == [3, 4]


def test_an_embedding_head_keeps_the_encoder_frozen_and_trains_in_one_pass(ws, cache, tiny_bert, monkeypatch):
    monkeypatch.setenv("BYOSM_DEVICE", "cpu")
    spec = head_spec(tiny_bert)
    write_rows(ws / "data")
    meta, _ = fit(spec, ws / "data", ws / "run", log=lambda _: None)
    state = torch.load(ws / "run" / WEIGHTS)
    fresh = build_model(spec, get_modality(spec)).backbone.state_dict()
    assert all(torch.equal(state[f"backbone.{k}"], v) for k, v in fresh.items())
    assert len(meta["history"]) == 1 and meta["best_epoch"] == 0 and meta["dev"]["objective"] > 0.9


def test_an_embedding_head_is_scored_and_exported_like_a_finetuned_model(ws, cache, tiny_bert, load_runtime):
    spec = head_spec(tiny_bert)
    p = Project("isq").ensure()
    p.save_spec(spec)
    write_rows(p.data_dir_for(spec))
    build("isq", log=lambda _: None)
    assert p.promoted == "v1" and p.board()["v1"]["builtin"]["objective"] > 0.9
    out = export("isq", log=lambda _: None)
    assert json.loads((out / "manifest.json").read_text())["backbone"] == tiny_bert
    m = load_runtime(out)
    assert m.predict("what about the coffee?")["label"] == "question"
    assert m.predict("i like the garden.")["label"] == "statement"


def test_a_student_distilled_from_an_embedding_head_is_fine_tuned():
    student = student_spec(head_spec("hf:x"), "v1")
    assert student.train.method == "finetune" and student.backbone == TINY_TEXT and student.train.distill_from == "v1"
