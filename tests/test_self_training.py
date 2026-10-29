import numpy as np

from byosm.data.store import TRAIN_INPUTS, Dataset, record, write
from byosm.project import Project
from byosm.spec import Label, TaskSpec, TrainCfg
from byosm.train import teacher


def test_the_promoted_model_labels_only_the_pool_rows_it_is_sure_of(ws, monkeypatch):
    labels = [Label(name="statement", definition=""), Label(name="question", definition="")]
    spec = TaskSpec(name="tc", description="", modality="text", labels=labels, train=TrainCfg(self_train=0.9))
    p = Project("tc").ensure()
    p.save_spec(spec)
    p.save_experiment("v1", meta={"params": 1})
    p.promote("v1", "test")
    dd = p.data_dir_for(spec)
    train = [record("already in train", "statement", "t")]
    pool = [record(t, None, "hf:pool") for t in ("is it on?", "it is on.", "maybe on")]
    write(dd / "unlabeled.jsonl", pool + [record("already in train", None, TRAIN_INPUTS)])
    sure = {"is it on?": [0.05, 0.95], "it is on.": [0.97, 0.03], "maybe on": [0.6, 0.4]}
    monkeypatch.setattr(
        teacher, "_probs", lambda spec, p, run_dir, data_dir, rows, dev: [sure[r["input"]] for r in rows]
    )
    out = teacher.pseudo_labels(spec, train, Dataset(dd), dd, "cpu", lambda _: None)
    assert [(r["input"], r["label"], r["source"]) for r in out] == [
        ("is it on?", "question", teacher.PSEUDO),
        ("it is on.", "statement", teacher.PSEUDO),
    ]
    assert np.isclose(spec.train.self_train, 0.9)
