import json
import random
import re

import numpy as np
import pytest
import torch
import yaml

from byojl import llm
from byojl.data.audio_mix import Mixer, frame_activity
from byojl.jepa import sigreg
from byojl.metrics import calibrate, compute
from byojl.modalities.audio import TinyCRNN
from byojl.spec import Label, Objective, TaskSpec

TINY_BERT = "hf:google/bert_uncased_L-4_H-256_A-4"


def test_sigreg_prefers_isotropic_gaussian():
    torch.manual_seed(0)
    gauss = torch.randn(512, 64)
    collapsed = torch.randn(512, 1).repeat(1, 64) * 0.01
    stretched = torch.randn(512, 64) * torch.linspace(0.1, 3, 64)
    g = sigreg(gauss)
    assert g < sigreg(collapsed) and g < sigreg(stretched)


def test_threshold_calibration_meets_precision_constraint():
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="neg", definition=""), Label(name="pos", definition="")],
                    positive_label="pos", objective=Objective(metric="recall_at_precision", constraint=0.95))
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    score = np.clip(y * 0.35 + rng.normal(0.35, 0.15, 2000), 0, 1)
    probs = np.stack([1 - score, score], 1)
    thr = calibrate(spec, y, probs)
    m = compute(spec, y, probs, thr)
    assert m["precision"] >= 0.95 and m["recall"] > 0.3


def test_crnn_frames_align_with_hop():
    m = TinyCRNN(16000, 10.0)
    wav = torch.randn(2, 16000)
    feats, mask = m(wav, torch.tensor([16000, 8000]))
    assert feats.shape[1] == 100 and mask[0].sum() == 100 and mask[1].sum() == 50


def test_mixer_labels_follow_events():
    spec = TaskSpec(name="vad", description="", modality="audio", task_type="frame",
                    labels=[Label(name="silence", definition=""), Label(name="speech", definition="")], positive_label="speech", backbone="tiny-crnn")
    t = np.arange(16000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, 0.01, 32000).astype(np.float32)
    mx = Mixer(spec, [tone], [noise], seed=1)
    for _ in range(20):
        wav, frames = mx.clip()
        assert len(frames) == len(wav) // 160
    act = frame_activity(np.concatenate([np.zeros(1600, np.float32), tone[:3200]]), 160)
    assert act[:10].sum() == 0 and act[10:30].all()


class FakeLLM:
    """Deterministic stand-in for an LLM: question vs statement."""

    name = "fake"

    def __init__(self, **_):
        self.rng = random.Random(0)

    def json(self, system, prompt, schema):
        props = schema["properties"]
        if "modality" in props:
            return {"modality": "text", "task_type": "classify", "summary": "is it a question", "input_description": "short utterances",
                    "labels": [{"name": "statement", "definition": "not a question"}, {"name": "question", "definition": "asks something"}],
                    "positive_label": "question", "objective_metric": "macro_f1", "objective_target": 0.99, "objective_constraint": 0.99,
                    "languages": ["en"], "diversity_axes": [{"name": "topic", "values": ["food", "travel", "code"]}],
                    "hard_cases": ["rhetorical question"], "backbone": TINY_BERT, "latency_budget_ms": 5, "synthetic": True,
                    "hf_queries": [], "hf_candidates": [], "train_size": 120, "test_size": 60}
        if "labels" in props:  # judge
            items = re.findall(r"^\d+\. (.*)$", prompt, re.M)
            return {"labels": ["question" if s.endswith("?") else "statement" for s in items]}
        if "patterns" in props:
            return {"patterns": [{"name": "short", "explanation": "short inputs", "data_recipe": "very short inputs", "share": 1.0}], "suspect_gold_ids": []}
        k = int(re.search(r"Write (\d+)", prompt).group(1))
        words = ["pasta", "train", "python", "rain", "museum", "bug", "coffee", "flight", "merge", "garden"]
        out = []
        for _ in range(k):
            w = " ".join(self.rng.sample(words, self.rng.randint(2, 6)))
            q = self.rng.random() < 0.5
            out.append({"text": (f"what about {w}?" if q else f"i like {w}."), "label": "question" if q else "statement"})
        return {"examples": out}


@pytest.fixture
def ws(tmp_path, monkeypatch):
    llm.register_provider("fake", FakeLLM)
    llm._cache.clear()
    (tmp_path / "byojl.yaml").write_text(yaml.safe_dump({"llm": {"default": {"provider": "fake"}}, "concurrency": 2}))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_text_pipeline_end_to_end(ws):
    from byojl import evals, improve
    from byojl.config import Project
    from byojl.export import export
    from byojl.pipeline import build
    from byojl.spec import compile_spec

    spec = compile_spec("isq", "detect questions")
    spec.data.dev_size, spec.data.per_call = 40, 20
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 3, 10, 1
    spec.baselines = ["llm"]
    p = Project("isq").ensure()
    spec.save(p.spec_path)
    build("isq", log=lambda m: None)
    assert p.promoted == "v1"
    board = json.loads((p.root / "board.json").read_text())
    assert board["v1"]["builtin"]["objective"] > 0.9 and any(k.startswith("llm") for k in board)

    # improve -> apply -> second experiment reuses the cached data (training-only change)
    dg, props = improve.propose("isq", use_llm=False, log=lambda m: None)
    (p.root / "proposals.json").write_text(json.dumps({"from": "v1", "proposals": [{"id": "x", "why": "", "ops": [{"op": "set", "path": "train.epochs", "value": 2}]}]}))
    exp = improve.apply("isq", ["x"])
    assert exp == "v2"
    build("isq", exp, log=lambda m: None)
    assert set(p.experiments()) == {"v1", "v2"} and len(list((p.root / "data").iterdir())) == 1

    # export -> torch-free runtime with the JEPA-style API
    out = export("isq", log=lambda m: None)
    import importlib.util

    s = importlib.util.spec_from_file_location("byojl_rt", out / "byojl_rt.py")
    rt = importlib.util.module_from_spec(s)
    s.loader.exec_module(rt)
    m = rt.load(out)
    assert m.predict("what about the coffee?")["label"] == "question"
    assert m.encode("what about the coffee?").shape == (256,)


def test_crnn_streaming_matches_offline():
    torch.manual_seed(0)
    m = TinyCRNN(16000, 10.0).eval()
    wav = torch.randn(1, 16000) * 0.1
    full, _ = m(wav, torch.tensor([16000]))
    state, outs = None, []
    for i in range(0, 16000, 480):  # 30ms chunks
        f, state = m.stream(wav[:, i : i + 480], state)
        outs.append(f)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4)


def _make_pdf(path, pages):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    for title, lines in pages:
        c.setFont("Helvetica-Bold", 22)
        c.drawString(72, 720, title)
        c.setFont("Helvetica", 11)
        for i, l in enumerate(lines):
            c.drawString(72, 690 - 16 * i, l)
        c.showPage()
    c.save()


def test_document_fused_pipeline_and_export(ws):
    from byojl.config import Project
    from byojl.data.store import Dataset, record
    from byojl.export import export
    from byojl.pipeline import build
    from byojl.spec import DocumentCfg, EvalSuite, TaskSpec, TrainCfg

    p = Project("doc").ensure()
    spec = TaskSpec(name="doc", description="invoice vs letter", modality="document", backbone="doc",
                    labels=[Label(name="invoice", definition=""), Label(name="letter", definition="")],
                    document=DocumentCfg(route="fused", text_backbone=TINY_BERT, image_backbone="tiny-cnn", max_pages=3, image_size=64, dim=64, ocr="never"),
                    train=TrainCfg(epochs=3, pretrain_steps=5, batch_size=8, lr_backbone=1e-4, pool="attention"), evals=[EvalSuite(name="builtin", primary=True)])
    spec.data.synthetic = False
    spec.save(p.spec_path)
    dd = p.data_dir_for(spec)
    (dd / "pdf").mkdir(parents=True)
    rng = random.Random(0)
    splits = {"train": [], "dev": [], "test": []}
    for i in range(60):
        lab = "invoice" if i % 2 else "letter"
        pages = [("INVOICE #%d" % i, ["Qty  Item  Price", "Total due: $%d" % rng.randint(10, 999)])] if lab == "invoice" else [("Dear Sir or Madam,", ["I am writing to you about", "Kind regards"])]
        pages += [("Appendix", ["terms and conditions"])] * rng.randint(0, 2)
        f = dd / "pdf" / f"{i}.pdf"
        _make_pdf(f, pages)
        splits["train" if i < 40 else ("dev" if i < 50 else "test")].append(record(f"pdf/{i}.pdf", lab, "test"))
    for s, rows in splits.items():
        Dataset(dd).save(s, rows)
    build("doc", log=lambda m: None)
    board = json.loads((p.root / "board.json").read_text())
    assert board["v1"]["builtin"]["objective"] > 0.8

    out = export("doc", log=lambda m: None)
    import importlib.util

    s = importlib.util.spec_from_file_location("byojl_rt", out / "byojl_rt.py")
    rt = importlib.util.module_from_spec(s)
    s.loader.exec_module(rt)
    m = rt.load(out)
    pred = m.predict(str(dd / "pdf" / "45.pdf"))
    assert pred["label"] in ("invoice", "letter") and m.encode(str(dd / "pdf" / "45.pdf")).shape == (64,)
