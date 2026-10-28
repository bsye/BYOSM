import random

import numpy as np
import pytest

from byosm.data.hf_source.rows import _tag_sequence
from byosm.data.store import write
from byosm.export import export
from byosm.metrics import bootstrap, compute, objective_of_units, units
from byosm.pipeline import build
from byosm.project import Project
from byosm.rt import OUTSIDE, bio, spans
from byosm.spec import Label, Objective, TaskSpec
from byosm.text import markup_tags

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
PEOPLE = ["alice", "bob", "carol smith", "dan jones"]
PLACES = ["paris", "rome", "new york", "san diego"]


def tagging_spec(name="ner"):
    spec = TaskSpec(
        name=name,
        description="find people and places",
        modality="text",
        task_type="frame",
        labels=[Label(name=t, definition=t) for t in TAGS],
        objective=Objective(metric="span_f1", target=0.99),
        backbone="hf:google/bert_uncased_L-4_H-256_A-4",
    )
    spec.data.synthetic = False
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 6, 0, 0
    return spec


def tagged(words, kind):
    parts = words.split()
    return parts, [f"B-{kind}"] + [f"I-{kind}"] * (len(parts) - 1)


def sentence(rng):
    who, wt = tagged(rng.choice(PEOPLE), "PER")
    where, pt = tagged(rng.choice(PLACES), "LOC")
    words = [*who, "flew", "to", *where, "today"]
    return {"input": " ".join(words), "label": [*wt, "O", "O", *pt, "O"], "source": "test", "id": str(rng.random())}


def test_spans_follow_bio_and_bare_types():
    tags = ["B-PER", "I-PER", "O", "B-LOC", "B-LOC", "I-ORG", "PER"]
    assert spans(tags) == [("PER", 0, 2), ("LOC", 3, 4), ("LOC", 4, 5), ("ORG", 5, 6), ("PER", 6, 7)]


def test_span_f1_counts_whole_spans_and_bootstraps_per_sentence():
    spec = tagging_spec()
    y = np.array([1, 2, 0, 3, 0, 1, 0])
    half = np.array([1, 0, 0, 3, 0, 1, 0])
    probs = np.eye(len(TAGS))[half]
    m = compute(spec, y, probs, None, [5, 2])
    assert m["span_precision"] == pytest.approx(2 / 3) and m["objective"] == pytest.approx(2 / 3)
    good, bad = units(5, y, y, [5, 2], spec.label_names), units(5, y, half, [5, 2], spec.label_names)
    assert objective_of_units(spec, good) == 1.0 and bootstrap(spec, good, bad)["delta"] > 0


def test_sequence_rows_map_tag_ids_to_our_labels():
    spec = tagging_spec()
    names = ["O", "B-PER", "I-PER", "B-MISC"]
    assert _tag_sequence([1, 2, 3, 0], names, {}, spec) == ["B-PER", "I-PER", "O", "O"]
    assert _tag_sequence([1, 3], names, {"B-PER": "B-PER"}, spec) == ["B-PER", "O"]


@pytest.mark.slow
def test_bracket_markup_becomes_words_and_bio_tags():
    words, tags = markup_tags("a [x : b c] d [y: e]")
    assert words == ["a", "b", "c", "d", "e"]
    assert tags == [OUTSIDE, *bio("x", 2), OUTSIDE, *bio("y", 1)]
    assert spans(tags) == [("x", 1, 3), ("y", 4, 5)]


def test_tagging_pipeline_end_to_end(ws, load_runtime):
    spec = tagging_spec()
    p = Project("ner").ensure()
    p.save_spec(spec)
    rng = random.Random(0)
    dd = p.data_dir_for(spec)
    for split, n in (("train", 300), ("dev", 40), ("test", 40)):
        write(dd / f"{split}.jsonl", [sentence(rng) for _ in range(n)])
    build("ner", log=lambda _: None)
    assert p.board()["v1"]["builtin"]["objective"] > 0.9
    out = load_runtime(export("ner", log=lambda _: None)).predict("carol smith flew to new york today")
    assert [(e["label"], e["text"]) for e in out["entities"]] == [("PER", "carol smith"), ("LOC", "new york")]
    assert len(out["confidence"]) == len(out["words"]) and all(0 < c <= 1 for c in out["confidence"])
