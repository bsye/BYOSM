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
from byosm.machine import backbones
from byosm.modalities import get_modality
from byosm.pipeline import build
from byosm.project import WEIGHTS, Project
from byosm.spec import Label, TaskSpec, TrainCfg
from byosm.train import build_model, fit
from byosm.train.contrastive import DIFFERENT, MAX_PAIRS, SAME, contrastive, pairs
from byosm.train.embedding import ABSENT_LOGIT, logistic_weights, pooled
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


def label_gap(emb, y):
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim, same = unit @ unit.T, np.equal.outer(y, y)
    return sim[same & ~np.eye(len(y), dtype=bool)].mean() - sim[~same].mean()


def test_pairs_mix_same_and_different_labels_and_never_pair_a_row_with_itself():
    labels = [0, 0, 0, 1, 1, 2]
    found = pairs(labels, 4, random.Random(0))
    assert all(a != b and (labels[a] == labels[b]) == (t == SAME) for a, b, t in found)
    per_row = {i: [t for a, _, t in found if a == i] for i in range(len(labels))}
    assert per_row[0].count(SAME) == per_row[0].count(DIFFERENT) == 2
    assert per_row[5] == [DIFFERENT, DIFFERENT]


def test_pairs_stay_under_the_cap_on_big_datasets():
    labels = [i % 7 for i in range(3000)]
    assert len(pairs(labels, 40, random.Random(0))) <= MAX_PAIRS


def test_contrastive_training_pulls_rows_of_one_label_together(tiny_bert):
    spec = head_spec(tiny_bert, method="setfit", pairs_per_row=8, batch_size=16, lr_pairs=1e-4)
    mod = get_modality(spec)
    model = build_model(spec, mod)
    rng = random.Random(0)
    rows = [utterance(rng) for _ in range(32)]
    raws, y = [r["input"] for r in rows], [spec.label_index[r["label"]] for r in rows]
    before = label_gap(pooled(model, mod, raws, "cpu"), y)
    contrastive(spec, mod, model, raws, y, random.Random(0), "cpu", lambda _: None)
    assert label_gap(pooled(model, mod, raws, "cpu"), y) > before


@pytest.mark.parametrize("method", ["embedding-head", "setfit"])
def test_only_setfit_moves_the_encoder_and_both_train_in_one_pass(ws, cache, tiny_bert, monkeypatch, method):
    monkeypatch.setenv("BYOSM_DEVICE", "cpu")
    spec = head_spec(tiny_bert, method=method, pairs_per_row=10)
    write_rows(ws / "data")
    meta, _ = fit(spec, ws / "data", ws / "run", log=lambda _: None)
    state = torch.load(ws / "run" / WEIGHTS)
    fresh = build_model(spec, get_modality(spec)).backbone.state_dict()
    assert all(torch.equal(state[f"backbone.{k}"], v) for k, v in fresh.items()) == (method == "embedding-head")
    assert len(meta["history"]) == 1 and meta["best_epoch"] == 0 and meta["dev"]["objective"] > 0.9


@pytest.mark.parametrize("method", ["embedding-head", "setfit"])
def test_an_embedding_method_is_scored_and_exported_like_a_finetuned_model(ws, cache, tiny_bert, load_runtime, method):
    spec = head_spec(tiny_bert, method=method, pairs_per_row=10)
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


def test_the_machine_page_measures_fine_tuning_speed_without_embedding_heads(ws):
    p = Project("isq").ensure()
    p.save_spec(head_spec(TINY_TEXT, method="finetune"))
    meta = {"train_seconds": 10.0, "n_train": 100, "history": [{}, {}], "params": 11_000_000}
    p.save_experiment("v1", spec=head_spec(TINY_TEXT, method="finetune"), meta=meta)
    p.save_experiment("v2", spec=head_spec(TINY_TEXT), meta={**meta, "train_seconds": 0.1, "history": [{}]})
    assert [(b["backbone"], b["rows_per_second"], b["runs"]) for b in backbones(ws)] == [(TINY_TEXT, 20.0, 1)]
