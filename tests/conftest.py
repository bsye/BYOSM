import importlib.util
import random

import numpy as np
import pytest
import torch
import yaml

from byosm import llm

TINY_BERT = "hf:google/bert_uncased_L-4_H-256_A-4"
SEED = 0

SPEC_REPLY = {
    "name": "Question detector",
    "title": "Question detector",
    "modality": "text",
    "task_type": "classify",
    "summary": "is it a question",
    "input_description": "short utterances",
    "labels": [
        {"name": "statement", "definition": "not a question"},
        {"name": "question", "definition": "asks something"},
    ],
    "positive_label": "question",
    "objective_metric": "macro_f1",
    "objective_target": 0.99,
    "objective_constraint": 0.99,
    "languages": ["en"],
    "diversity_axes": [{"name": "topic", "values": ["food", "travel", "code"]}],
    "hard_cases": ["rhetorical question"],
    "backbone": TINY_BERT,
    "latency_budget_ms": 5,
    "synthetic": True,
    "hf_queries": [],
    "hf_candidates": [],
    "train_size": 120,
    "test_size": 60,
}

WORDS = ["pasta", "train", "python", "rain", "museum", "bug", "coffee", "flight", "merge", "garden"]


def numbered_items(prompt):
    items = []
    for line in prompt.splitlines():
        head, sep, rest = line.partition(". ")
        if sep and head.isdigit():
            items.append(rest)
    return items


def item_blocks(prompt):
    lines = prompt.splitlines()
    return [
        lines[i + 1]
        for i, line in enumerate(lines[:-1])
        if line.startswith("Item ") and line.endswith(":") and line[5:-1].isdigit()
    ]


def requested_count(prompt):
    for chunk in prompt.split("Write ")[1:]:
        digits = chunk[: len(chunk) - len(chunk.lstrip("0123456789"))]
        if digits:
            return int(digits)
    raise ValueError("no example count in prompt")


def kind(text):
    return "question" if text.endswith("?") else "statement"


class FakeLLM:
    name = "fake"

    def __init__(self, **_):
        self.rng = random.Random(0)

    def json(self, system, prompt, schema):
        props = schema["properties"]
        if "modality" in props:
            return dict(SPEC_REPLY)
        if "confusable" in props:
            pair = {"a": "statement", "b": "question", "why": "rhetorical questions"}
            return {"confusable": [pair], "tips": [], "examples": [{"label": "question", "input": "is it on?"}]}
        if "why" in props:
            return {"label": "question", "why": "it asks something"}
        if "labels" in props:
            return self.label(prompt)
        if "patterns" in props:
            pattern = {"name": "short", "explanation": "short inputs", "data_recipe": "very short inputs", "share": 1.0}
            return {"patterns": [pattern], "suspect_gold_ids": []}
        if "rewrites" in props:
            k = requested_count(prompt)
            items = numbered_items(prompt)
            return {
                "rewrites": [{"n": i, "texts": [f"{s} (take {j})" for j in range(k)]} for i, s in enumerate(items, 1)]
            }
        return {"examples": [self.example() for _ in range(requested_count(prompt))]}

    def label(self, prompt):
        items = numbered_items(prompt)
        if items:
            return {"labels": [{"n": i, "label": kind(s)} for i, s in enumerate(items, 1)]}
        truth = [kind(s) for s in item_blocks(prompt)]
        flip = {"question": "statement", "statement": "question"}
        answers = [t if self.rng.random() < 0.75 else flip[t] for t in truth]
        return {"labels": [{"n": i, "label": t} for i, t in enumerate(answers, 1)]}

    def example(self):
        w = " ".join(self.rng.sample(WORDS, self.rng.randint(2, 6)))
        q = self.rng.random() < 0.5
        return {"text": (f"what about {w}?" if q else f"i like {w}."), "label": "question" if q else "statement"}


class Flat:
    def __init__(self, name, level=0.0):
        self.name, self.level, self.fits = name, level, []

    def fit(self, groups):
        self.fits.append(len(groups))

    def predict_many(self, histories, season, horizon, clocks=None):
        from byosm.series.metrics import QUANTILES

        return [np.full((len(QUANTILES), horizon), h[-1] + self.level) for h in histories]


def save_forecaster(ws, domain="demo", ours=0.9, rival=0.8, model="patch transformer"):
    from byosm.series.board import save_report
    from byosm.series.prediction import Options
    from byosm.series.specialist import NAIVE

    row = {
        NAIVE: {"mase": 1.0, "calib_mase": 1.0, "crps": 1.0},
        model: {"mase": ours, "calib_mase": ours, "crps": 0.5},
        "Timestral-5M": {"mase": rival, "crps": 0.4},
    }
    scores = {"calib": ours, "test": ours, "crps": 0.5, "ci": [ours - 0.01, ours + 0.01], "at": 1.0}
    version = {"name": "v1", "recipe": model, "effort": 0, **Options().model_dump(), **scores, "configs": {"a": row}}
    report = {"domain": domain, "at": 1.0, "configs": {"a": row}, "versions": [version], "promoted": "v1"}
    save_report(ws, report)
    return report


def import_runtime(out):
    s = importlib.util.spec_from_file_location("byosm_rt", out / "byosm_rt.py")
    rt = importlib.util.module_from_spec(s)
    s.loader.exec_module(rt)
    return rt.load(out)


PAIRED_ROWS = 200


def paired(right: float) -> dict:
    hits = round(right * PAIRED_ROWS)
    split = {"y": [1] * PAIRED_ROWS, "pred": [1] * hits + [0] * (PAIRED_ROWS - hits)}
    return {"units": {"calib": split, "test": split}}


def save_board(p, board):
    for pid, by_suite in board.items():
        for suite, data in by_suite.items():
            p.save_result(pid, suite, data)


def promoted_specialist(
    name="qa", n_test=500, n_calib=200, rivals=("jev",), labels=("statement", "question"), positive=None
):
    from byosm.project import Project
    from byosm.spec import Label, TaskSpec

    p = Project(name).ensure()
    spec = TaskSpec(
        name=name,
        description="is it a question",
        modality="text",
        labels=[Label(name=n, definition=n) for n in labels],
        positive_label=positive,
        baselines=list(rivals),
    )
    p.save_spec(spec)
    p.exp_dir("v1").mkdir(parents=True)
    p.save_experiment("v1", meta={})
    p.promote("v1", "test")
    save_board(p, {"v1": {"builtin": {"kind": "experiment", "objective": 0.9, "n_test": n_test, "n_calib": n_calib}}})
    return p


def with_train_rows(p, per_label):
    from byosm.data.store import Dataset, record

    spec = p.spec()
    rows = [record(f"{lab} example {i}", lab, "x") for lab in spec.label_names for i in range(per_label)]
    Dataset(p.data_dir_for(spec)).save("train", rows)


def write_pdf(path, pages):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    for title, lines in pages:
        c.setFont("Helvetica-Bold", 22)
        c.drawString(72, 720, title)
        c.setFont("Helvetica", 11)
        for i, line in enumerate(lines):
            c.drawString(72, 690 - 16 * i, line)
        c.showPage()
    c.save()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end build, skip with -m 'not slow'")


@pytest.fixture(autouse=True)
def no_web_research(monkeypatch):
    from byosm.data.hf_source import research

    monkeypatch.setattr(research, "research_config", lambda cfg: None)


@pytest.fixture(autouse=True)
def gpu_memory_is_free(monkeypatch):
    from byosm.train import gpu

    monkeypatch.setattr(gpu, "free_gb", lambda: float(gpu.GB_PER_TRAINING))


@pytest.fixture(autouse=True)
def fresh_shared_database():
    import os

    from byosm.db.engine import URL_ENV, engine
    from byosm.db.models import Base

    if os.environ.get(URL_ENV):
        eng = engine(os.environ[URL_ENV])
        Base.metadata.drop_all(eng)
        Base.metadata.create_all(eng)


@pytest.fixture(autouse=True)
def seeded():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.fixture(scope="session")
def tiny_bert():
    return TINY_BERT


@pytest.fixture
def load_runtime():
    return import_runtime


@pytest.fixture
def make_pdf():
    return write_pdf


@pytest.fixture
def ws(tmp_path, monkeypatch):
    llm.register_provider("fake", FakeLLM)
    llm.clear_cache()
    (tmp_path / "byosm.yaml").write_text(yaml.safe_dump({"llm": {"default": {"provider": "fake"}}, "concurrency": 2}))
    monkeypatch.chdir(tmp_path)
    return tmp_path
